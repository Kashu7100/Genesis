import dataclasses
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

import genesis as gs
from genesis.engine.entities.rigid_entity import KinematicEntity
from genesis.engine.entities.rigid_entity.description import _balanced_variant_mapping
from genesis.engine.materials.base import Material
from genesis.engine.scene import Scene
from genesis.options import (
    BaseCouplerOptions,
    FEMOptions,
    KinematicOptions,
    MPMOptions,
    PBDOptions,
    ProfilingOptions,
    RigidOptions,
    SceneOptions,
    SimOptions,
    SPHOptions,
    ViewerOptions,
    VisOptions,
)
from genesis.options.entities import Spec, get_entity_spec
from genesis.options.morphs import Morph
from genesis.options.renderers import RendererOptions
from genesis.options.surfaces import Surface
from genesis.repr_base import RBC
from genesis.utils.misc import BuildableMixin, indices_to_mask

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidEntity


@dataclasses.dataclass(frozen=True)
class EnvironmentPartition:
    """The partition of the environments of a heterogeneous scene into groups, each group simulated by one scene.

    The environments of a group keep their order in the scene of the group: the environment of local index i in group g
    is 'groups_envs_idx[g][i]'.
    """

    groups_variants_idx: tuple[tuple[int, ...], ...]
    """The variant of each entity that each group carries, the entities in the order they were added."""
    groups_envs_idx: tuple[torch.Tensor, ...]
    """The environments of each group, in increasing order."""
    envs_group_idx: torch.Tensor
    """The group of each environment."""
    envs_local_idx: torch.Tensor
    """The index of each environment in the scene of its group."""

    def select(self, envs_idx=None) -> list[tuple[int, torch.Tensor, torch.Tensor | None]]:
        """Split a selection of environments by group.

        'envs_idx' selects environments of the heterogeneous scene by index, like the 'envs_idx' argument of the
        accessors of a scene, None selecting them all. The result holds one entry per group with at least one selected
        environment: the index of the group, the positions of its selected environments in the selection, and their
        indices in the scene of the group, None if 'envs_idx' is None.
        """
        if envs_idx is None:
            return [(i_group, rows, None) for i_group, rows in enumerate(self.groups_envs_idx)]
        (envs_mask,) = indices_to_mask(envs_idx, boolean_mask=False)
        envs_group_idx = self.envs_group_idx[envs_mask]
        envs_local_idx = self.envs_local_idx[envs_mask]
        # Sorting by group gathers the selected environments of each group in one block, so a single read back of the
        # counts splits the selection without synchronizing once per group.
        envs_pos = torch.argsort(envs_group_idx, stable=True)
        groups_n_envs = torch.bincount(envs_group_idx, minlength=len(self.groups_envs_idx)).tolist()
        selection = []
        i_start = 0
        for i_group, n_envs in enumerate(groups_n_envs):
            if n_envs > 0:
                rows = envs_pos[i_start : i_start + n_envs]
                selection.append((i_group, rows, envs_local_idx[rows]))
            i_start += n_envs
        return selection


class HeterogeneousEntity(BuildableMixin, RBC):
    """An entity of a heterogeneous scene, simulated in each group scene by an entity of the variant the group carries.

    Its accessors read and write all the environments at once, or those 'envs_idx' selects by index in the heterogeneous
    scene, each group scene receiving the environments it holds. The quantities whose size depends on the variant (the
    qpos, the degrees of freedom and the links) span the largest variant, an environment carrying the leading entries of
    this layout: a getter reads zero where the variant has none, and a setter leaves these entries out. Anything else is
    reached through the entity of each group, 'entities'.
    """

    def __init__(self, spec: Spec, morphs: tuple[Morph, ...]):
        self._spec = spec
        self._morphs = morphs
        self._entities: tuple["RigidEntity", ...] = ()
        self._partition: EnvironmentPartition | None = None
        self._groups_n_qs: tuple[int, ...] = ()
        self._groups_n_dofs: tuple[int, ...] = ()
        self._groups_n_links: tuple[int, ...] = ()

    @gs.assert_unbuilt
    def build(self, entities: Sequence["RigidEntity"], partition: EnvironmentPartition) -> None:
        """Record the entity of each group scene, 'entities' following the order of the groups in 'partition'."""
        self._entities = tuple(entities)
        self._partition = partition
        # Only the kinematic entities have degrees of freedom, and the accessors wrapped here are theirs alone
        if isinstance(self._entities[0], KinematicEntity):
            self._groups_n_qs = tuple(entity.n_qs for entity in self._entities)
            self._groups_n_dofs = tuple(entity.n_dofs for entity in self._entities)
            self._groups_n_links = tuple(entity.n_links for entity in self._entities)
        self._is_built = True

    def _gather(self, getters: Sequence[Callable], envs_idx, size: int | None) -> torch.Tensor:
        selection = self._partition.select(envs_idx)
        values = [(rows, getters[i_group](envs_idx=envs_idx_local)) for i_group, rows, envs_idx_local in selection]
        n_envs = sum(rows.shape[0] for rows, _ in values)
        value = values[0][1]
        shape = (n_envs, *value.shape[1:]) if size is None else (n_envs, size, *value.shape[2:])
        tensor = torch.zeros(shape, dtype=value.dtype, device=gs.device)
        for rows, value_group in values:
            if size is None:
                tensor[rows] = value_group
            else:
                tensor[rows, : value_group.shape[1]] = value_group
        return tensor

    def _scatter(self, setters: Sequence[Callable], value, envs_idx, groups_size: tuple[int, ...] | None) -> None:
        value = torch.as_tensor(value, dtype=gs.tc_float, device=gs.device)
        selection = self._partition.select(envs_idx)
        if groups_size is not None and value.ndim > 0 and value.shape[-1] != max(groups_size):
            gs.raise_exception(f"Expected {max(groups_size)} entries along the last axis, got {value.shape[-1]}.")
        n_envs = sum(rows.shape[0] for _, rows, _ in selection)
        if value.ndim > 1 and value.shape[0] != n_envs:
            gs.raise_exception(f"Expected one row per selected environment ({n_envs}), got {value.shape[0]}.")
        for i_group, rows, envs_idx_local in selection:
            value_group = value[rows] if value.ndim > 1 else value
            if groups_size is not None and value.ndim > 0:
                value_group = value_group[..., : groups_size[i_group]]
            setters[i_group](value_group, envs_idx=envs_idx_local)

    # ------------------------------------------------------------------------------------
    # ------------------------------------ getters ---------------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def get_qpos(self, envs_idx=None) -> torch.Tensor:
        """Get the qpos of every environment, padded with zeros to 'n_qs'."""
        return self._gather([entity.get_qpos for entity in self._entities], envs_idx, self.n_qs)

    @gs.assert_built
    def get_dofs_position(self, envs_idx=None) -> torch.Tensor:
        """Get the position of the degrees of freedom of every environment, padded with zeros to 'n_dofs'."""
        return self._gather([entity.get_dofs_position for entity in self._entities], envs_idx, self.n_dofs)

    @gs.assert_built
    def get_dofs_velocity(self, envs_idx=None) -> torch.Tensor:
        """Get the velocity of the degrees of freedom of every environment, padded with zeros to 'n_dofs'."""
        return self._gather([entity.get_dofs_velocity for entity in self._entities], envs_idx, self.n_dofs)

    @gs.assert_built
    def get_dofs_force(self, envs_idx=None) -> torch.Tensor:
        """Get the smooth force on the degrees of freedom over the last step (see 'RigidEntity.get_dofs_force').

        The result is padded with zeros to 'n_dofs'.
        """
        return self._gather([entity.get_dofs_force for entity in self._entities], envs_idx, self.n_dofs)

    @gs.assert_built
    def get_dofs_control_force(self, envs_idx=None) -> torch.Tensor:
        """Get the control force the last step applied to the degrees of freedom, padded with zeros to 'n_dofs'."""
        return self._gather([entity.get_dofs_control_force for entity in self._entities], envs_idx, self.n_dofs)

    @gs.assert_built
    def get_pos(self, envs_idx=None) -> torch.Tensor:
        """Get the position of the base link of every environment."""
        return self._gather([entity.get_pos for entity in self._entities], envs_idx, size=None)

    @gs.assert_built
    def get_quat(self, envs_idx=None) -> torch.Tensor:
        """Get the quaternion of the base link of every environment."""
        return self._gather([entity.get_quat for entity in self._entities], envs_idx, size=None)

    @gs.assert_built
    def get_vel(self, envs_idx=None) -> torch.Tensor:
        """Get the linear velocity of the base link of every environment."""
        return self._gather([entity.get_vel for entity in self._entities], envs_idx, size=None)

    @gs.assert_built
    def get_ang(self, envs_idx=None) -> torch.Tensor:
        """Get the angular velocity of the base link of every environment."""
        return self._gather([entity.get_ang for entity in self._entities], envs_idx, size=None)

    @gs.assert_built
    def get_links_pos(self, envs_idx=None) -> torch.Tensor:
        """Get the position of the links of every environment, padded with zeros to 'n_links'."""
        return self._gather([entity.get_links_pos for entity in self._entities], envs_idx, self.n_links)

    @gs.assert_built
    def get_links_quat(self, envs_idx=None) -> torch.Tensor:
        """Get the quaternion of the links of every environment, padded with zeros to 'n_links'."""
        return self._gather([entity.get_links_quat for entity in self._entities], envs_idx, self.n_links)

    @gs.assert_built
    def get_mass(self, envs_idx=None) -> torch.Tensor:
        """Get the total mass of the entity in every environment, in kg, over its links not fixed to the world."""
        return self._gather([entity.get_mass for entity in self._entities], envs_idx, size=None)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- setters ----------------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def set_qpos(self, qpos, envs_idx=None) -> None:
        """Set the qpos, zeroing the velocity of the degrees of freedom.

        'qpos' is a scalar, a vector of 'n_qs' entries shared by every selected environment, or one such vector per
        selected environment. The entries beyond the qpos of the variant of an environment are left out.
        """
        self._scatter([entity.set_qpos for entity in self._entities], qpos, envs_idx, self._groups_n_qs)

    @gs.assert_built
    def set_dofs_position(self, position, envs_idx=None) -> None:
        """Set the position of the degrees of freedom, zeroing their velocity.

        'position' is a scalar, a vector of 'n_dofs' entries shared by every selected environment, or one such vector
        per selected environment. The entries beyond the degrees of freedom of the variant of an environment are left
        out, as for every setter of degrees of freedom of this entity.
        """
        self._scatter([entity.set_dofs_position for entity in self._entities], position, envs_idx, self._groups_n_dofs)

    @gs.assert_built
    def set_dofs_velocity(self, velocity, envs_idx=None) -> None:
        """Set the velocity of the degrees of freedom (see 'set_dofs_position' for 'velocity')."""
        self._scatter([entity.set_dofs_velocity for entity in self._entities], velocity, envs_idx, self._groups_n_dofs)

    @gs.assert_built
    def set_dofs_kp(self, kp, envs_idx=None) -> None:
        """Set the positional gains of the controller of the degrees of freedom (see 'set_dofs_position' for 'kp').

        Selecting environments requires 'RigidOptions.batch_dofs_info', as for an entity of a scene.
        """
        self._scatter([entity.set_dofs_kp for entity in self._entities], kp, envs_idx, self._groups_n_dofs)

    @gs.assert_built
    def set_dofs_kv(self, kv, envs_idx=None) -> None:
        """Set the velocity gains of the controller of the degrees of freedom (see 'set_dofs_kp' for 'kv')."""
        self._scatter([entity.set_dofs_kv for entity in self._entities], kv, envs_idx, self._groups_n_dofs)

    @gs.assert_built
    def control_dofs_position(self, position, envs_idx=None) -> None:
        """Set the target position of the controller of the degrees of freedom (see 'set_dofs_position')."""
        setters = [entity.control_dofs_position for entity in self._entities]
        self._scatter(setters, position, envs_idx, self._groups_n_dofs)

    @gs.assert_built
    def control_dofs_velocity(self, velocity, envs_idx=None) -> None:
        """Set the target velocity of the controller of the degrees of freedom (see 'set_dofs_position')."""
        setters = [entity.control_dofs_velocity for entity in self._entities]
        self._scatter(setters, velocity, envs_idx, self._groups_n_dofs)

    @gs.assert_built
    def control_dofs_force(self, force, envs_idx=None) -> None:
        """Set the motor force of the degrees of freedom (see 'set_dofs_position' for 'force')."""
        self._scatter([entity.control_dofs_force for entity in self._entities], force, envs_idx, self._groups_n_dofs)

    @gs.assert_built
    def set_pos(self, pos, envs_idx=None) -> None:
        """Set the position of the base link, zeroing the velocity of the degrees of freedom.

        'pos' is one position shared by every selected environment, or one position per selected environment.
        """
        self._scatter([entity.set_pos for entity in self._entities], pos, envs_idx, groups_size=None)

    @gs.assert_built
    def set_quat(self, quat, envs_idx=None) -> None:
        """Set the quaternion of the base link, zeroing the velocity of the degrees of freedom (see 'set_pos')."""
        self._scatter([entity.set_quat for entity in self._entities], quat, envs_idx, groups_size=None)

    # ------------------------------------------------------------------------------------
    # ---------------------------------- properties --------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def spec(self) -> Spec:
        """The spec of the entity, each group scene replacing its morph with that of the variant the group carries."""
        return self._spec

    @property
    def morphs(self) -> tuple[Morph, ...]:
        """The morphs of the entity, one per variant."""
        return self._morphs

    @gs.assert_built
    @property
    def entities(self) -> tuple["RigidEntity", ...]:
        """The entity of each group scene, in the order of the groups of the partition of the scene."""
        return self._entities

    @gs.assert_built
    @property
    def n_qs(self) -> int:
        """The number of entries of the qpos of the largest variant."""
        return max(self._groups_n_qs)

    @gs.assert_built
    @property
    def n_dofs(self) -> int:
        """The number of degrees of freedom of the largest variant."""
        return max(self._groups_n_dofs)

    @gs.assert_built
    @property
    def n_links(self) -> int:
        """The number of links of the largest variant."""
        return max(self._groups_n_links)

    @gs.assert_built
    @property
    def envs_n_qs(self) -> torch.Tensor:
        """The number of entries of the qpos of the variant each environment carries."""
        return torch.tensor(self._groups_n_qs, dtype=gs.tc_int, device=gs.device)[self._partition.envs_group_idx]

    @gs.assert_built
    @property
    def envs_n_dofs(self) -> torch.Tensor:
        """The number of degrees of freedom of the variant each environment carries."""
        return torch.tensor(self._groups_n_dofs, dtype=gs.tc_int, device=gs.device)[self._partition.envs_group_idx]

    @gs.assert_built
    @property
    def envs_n_links(self) -> torch.Tensor:
        """The number of links of the variant each environment carries."""
        return torch.tensor(self._groups_n_links, dtype=gs.tc_int, device=gs.device)[self._partition.envs_group_idx]

    @gs.assert_built
    @property
    def active_dofs_mask(self) -> torch.Tensor:
        """The mask of the degrees of freedom each environment carries, over the 'n_dofs' of the padded layout."""
        return torch.arange(self.n_dofs, device=gs.device) < self.envs_n_dofs[:, None]


@gs.assert_initialized
class HeterogeneousScene(BuildableMixin, RBC):
    """A batch of parallel environments whose entities may differ in structure from one environment to another.

    An entity added with several morphs carries one of them per environment, the environments being split into
    contiguous blocks of nearly equal size, as for a heterogeneous entity of a scene. The environments are partitioned
    into groups by the combination of variants they carry, and each group is an ordinary scene holding one variant of
    every entity, built with as many environments as the group holds. The variants may thus differ in anything a scene
    accepts: the number of links and degrees of freedom, the joint types, a fixed or a free base, the equality
    constraints. Each group scene is simulated at its own size, so an environment takes the memory and the time of the
    variants it carries. Each group scene launches its own kernels at every step, so the time of a step also grows with
    the number of groups, at most one plus the sum over the entities of their number of variants minus one.

    The options are those of 'gs.Scene', and each group scene receives a copy of them. The interactive viewer shows the
    environments of the first group alone. Cameras, sensors, recorders, emitters and force fields are added to a scene
    before it is built, so a heterogeneous scene, whose group scenes come to exist as it builds, offers none of them.
    After build, the scene of each group ('scenes') and the entity of each group ('HeterogeneousEntity.entities') give
    access to everything a built scene offers.
    """

    def __init__(
        self,
        sim_options: SimOptions | None = None,
        rigid_options: RigidOptions | None = None,
        kinematic_options: KinematicOptions | None = None,
        mpm_options: MPMOptions | None = None,
        sph_options: SPHOptions | None = None,
        fem_options: FEMOptions | None = None,
        pbd_options: PBDOptions | None = None,
        coupler_options: BaseCouplerOptions | None = None,
        vis_options: VisOptions | None = None,
        viewer_options: ViewerOptions | None = None,
        profiling_options: ProfilingOptions | None = None,
        renderer: RendererOptions | None = None,
        show_viewer: bool | None = None,
        options: SceneOptions | None = None,
    ):
        individual_options = (
            sim_options,
            rigid_options,
            kinematic_options,
            mpm_options,
            sph_options,
            fem_options,
            pbd_options,
            coupler_options,
            vis_options,
            viewer_options,
            profiling_options,
            renderer,
        )
        if options is not None:
            if any(option is not None for option in individual_options):
                gs.raise_exception("`options` holds every option a scene is created with, so it is given alone.")
            self._options = options
        else:
            self._options = SceneOptions(
                sim=sim_options,
                rigid=rigid_options,
                kinematic=kinematic_options,
                mpm=mpm_options,
                sph=sph_options,
                fem=fem_options,
                pbd=pbd_options,
                coupler=coupler_options,
                vis=vis_options,
                viewer=viewer_options,
                profiling=profiling_options,
                renderer=renderer,
            )
        self._show_viewer = show_viewer
        self._entities: list[HeterogeneousEntity] = []
        self._scenes: tuple[Scene, ...] = ()
        self._partition: EnvironmentPartition | None = None

    @gs.assert_unbuilt
    def add_entity(
        self,
        morph: Morph | Iterable[Morph],
        material: Material | None = None,
        surface: Surface | None = None,
        visualize_contact: bool | None = None,
        name: str | None = None,
    ) -> HeterogeneousEntity:
        """Add an entity to every group scene, as 'gs.Scene.add_entity' does.

        Parameters
        ----------
        morph : gs.morphs.Morph | list[gs.morphs.Morph]
            The morph of the entity, or one morph per variant. Each environment carries one variant, the environments
            being split into contiguous blocks of nearly equal size, and the variants may differ in structure.
        material : gs.materials.Material | None, optional
            The material of the entity. If None, use ``gs.materials.Rigid()``.
        surface : gs.surfaces.Surface | None, optional
            The surface of the entity. If None, use ``gs.surfaces.Default()``.
        visualize_contact : bool | None, optional
            Whether the contacts of the entity are drawn as markers (see 'gs.Scene.add_entity'). If None, defaults to
            False.
        name : str | None, optional
            The name of the entity in every group scene. If None, each group scene generates one.

        Returns
        -------
        entity : HeterogeneousEntity
            The entity, whose accessors are available once the scene is built.
        """
        morphs = (morph,) if isinstance(morph, Morph) else tuple(morph)
        spec = get_entity_spec(morphs[0], material, surface, visualize_contact, name=name)
        entity = HeterogeneousEntity(spec, morphs)
        self._entities.append(entity)
        return entity

    @gs.assert_unbuilt
    def build(self, n_envs: int, env_spacing=(0.0, 0.0), n_envs_per_row: int | None = None, center_envs_at_origin=True):
        """Partition the environments into groups, then create and build the scene of each group.

        Parameters
        ----------
        n_envs : int
            The number of parallel environments, at least one.
        env_spacing : tuple of float, shape (2,)
            The spacing between adjacent environments of each group scene, for visualization only. The environments of
            different groups overlap.
        n_envs_per_row : int | None
            The number of environments per row of each group scene, for visualization only. If None, it is set to the
            square root of the number of environments of the group.
        center_envs_at_origin : bool
            Whether to put the center of the environments of each group scene at the origin, for visualization only.
        """
        if n_envs < 1:
            gs.raise_exception(f"A heterogeneous scene has at least one environment, got 'n_envs={n_envs}'.")

        envs_variants_idx = np.empty((n_envs, len(self._entities)), dtype=gs.np_int)
        for i_entity, entity in enumerate(self._entities):
            envs_variants_idx[:, i_entity] = _balanced_variant_mapping(len(entity.morphs), n_envs)
        groups_variants_idx, envs_group_idx = np.unique(envs_variants_idx, axis=0, return_inverse=True)

        envs_local_idx = np.empty((n_envs,), dtype=gs.np_int)
        groups_envs_idx, scenes, groups_entities = [], [], []
        for i_group, variants_idx in enumerate(groups_variants_idx):
            envs_idx = np.flatnonzero(envs_group_idx == i_group)
            envs_local_idx[envs_idx] = np.arange(len(envs_idx))
            groups_envs_idx.append(torch.as_tensor(envs_idx, device=gs.device))

            scene = Scene(
                options=self._options.model_copy(deep=True), show_viewer=self._show_viewer if i_group == 0 else False
            )
            groups_entities.append(
                [
                    scene.add_entity(spec=entity.spec.model_copy(update={"morph": entity.morphs[i_variant]}))
                    for entity, i_variant in zip(self._entities, variants_idx)
                ]
            )
            scene.build(
                n_envs=len(envs_idx),
                env_spacing=env_spacing,
                n_envs_per_row=n_envs_per_row,
                center_envs_at_origin=center_envs_at_origin,
            )
            scenes.append(scene)

        self._scenes = tuple(scenes)
        self._partition = EnvironmentPartition(
            groups_variants_idx=tuple(map(tuple, groups_variants_idx.tolist())),
            groups_envs_idx=tuple(groups_envs_idx),
            envs_group_idx=torch.as_tensor(envs_group_idx, device=gs.device),
            envs_local_idx=torch.as_tensor(envs_local_idx, device=gs.device),
        )
        for entity, entities in zip(self._entities, zip(*groups_entities)):
            entity.build(entities, self._partition)
        self._is_built = True

    @gs.assert_built
    def step(self) -> None:
        """Run one simulation step of every group scene."""
        for scene in self._scenes:
            scene.step()

    @gs.assert_built
    def reset(self, envs_idx=None) -> None:
        """Reset the environments 'envs_idx' selects by index to their initial state, all of them if None."""
        for i_group, _, envs_idx_local in self._partition.select(envs_idx):
            self._scenes[i_group].reset(envs_idx=envs_idx_local)

    def destroy(self) -> None:
        """Destroy every group scene, releasing its memory."""
        for scene in self._scenes:
            scene.destroy()
        self._scenes = ()
        self._is_built = False

    @property
    def options(self) -> SceneOptions:
        """The options each group scene receives a copy of."""
        return self._options

    @property
    def entities(self) -> tuple[HeterogeneousEntity, ...]:
        """The entities, in the order they were added."""
        return tuple(self._entities)

    @property
    def is_built(self) -> bool:
        """Whether the scene has been built."""
        return self._is_built

    @gs.assert_built
    @property
    def n_envs(self) -> int:
        """The number of parallel environments."""
        return len(self._partition.envs_group_idx)

    @gs.assert_built
    @property
    def scenes(self) -> tuple[Scene, ...]:
        """The scene of each group, in the order of the groups of 'partition'."""
        return self._scenes

    @gs.assert_built
    @property
    def partition(self) -> EnvironmentPartition:
        """The partition of the environments into groups."""
        return self._partition
