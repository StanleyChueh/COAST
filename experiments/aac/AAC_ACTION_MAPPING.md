# GR00T N1.5 RoboCasa action dims → AAC entropy groups

Applies to the COAST checkpoint `robocasa365_checkpoints/gr00t_n1-5/multitask_learning/checkpoint-120000` and to upstream's `single_panda_gripper` data config. Both have the same action keys in the same concat order (`ROBOCASA_ACTION_KEYS` in `groot_env/groot_adapter.py`; `SinglePandaGripperDataConfig.action_keys` in gr00t-multi-sample).

Two index spaces are involved:
- the **normalized** model output `normalized_action` (N, 16, 32). Concat order, then zero padding up to 32. Per-key normalization: `min_max` maps to [-1, 1], `binary` stays in {0, 1}-ish logits.
- the **unnormalized** per-key dict that the client executes.

| Normalized idx | Client 12-D idx | Key | Normalization | AAC use (upstream, reproduced) |
|---|---|---|---|---|
| 0–2 | 0–2 | `action.end_effector_position` (Δx, Δy, Δz) | min_max | **E_t**: Gaussian entropy, d=3, on *normalized* dims 0:3. **m_t**: ‖Σ‖ of *unnormalized* candidate-0 values |
| 3–5 | 3–5 | `action.end_effector_rotation` (axis-angle Δ) | min_max | **E_r**: Gaussian entropy, d=3, on *normalized* dims 3:6. **m_r**: ‖axis-angle(∏ quat)‖ of *unnormalized* candidate-0 values |
| 6 | 6 | `action.gripper_close` | binary | **E_g**: Bernoulli entropy on the *unnormalized* {0,1} value across N. **m_g**: 0.2·1[toggle] on candidate 0 |
| 7–10 | 7–10 | `action.base_motion` (base x, y, yaw; torso) | min_max | **not used** by entropy or magnitude |
| 11 | 11 | `action.control_mode` (arm vs base) | binary | **not used** |
| 12–31 | — | padding (`max_action_dim=32`) | — | not used |

Evidence:
- Upstream `action_entropy_gausssian_bernoulli` reads only `normalized_action[:, :, :3]`, `normalized_action[:, :, 3:6]` and `action.gripper_close`. `action_magnitude` reads only `action.end_effector_position`, `action.end_effector_rotation` and `action.gripper_close`. `normalized_action[:, :, 6:7]` is assigned (`gripper_close_p`) but never used. `select_chunk_id` ("mean") would use `[:, :, :7]`, but RoboCasa runs use selector "0".
- The unnormalized gripper has shape (N, 16) upstream because of the global `np.squeeze`. COAST's adapter yields (N, 16, 1). `aac.build_upstream_pred_action_dict` squeezes the trailing axis so that `np.diff` in `action_magnitude` runs over time; otherwise toggles would never be detected (see `test_unsqueezed_gripper_would_hide_toggles`).

## Execution-side difference (documented, not changed)

The upstream AAC RoboCasa client **executes only arm + gripper**. It fills `action_dict["right"]` (pos + rot) and `action_dict["right_gripper"] = (g − 0.5)·2`, then calls `robot.create_action_vector`, which leaves base, torso and base_mode at zero. So upstream discards dims 7–11 at execution.

COAST's client executes all 12 dims through `robocasa.utils.env_utils.convert_action` and the gym wrapper's `unmap_action`. This is how the RoboCasa365 checkpoint is evaluated by its authors, and how COAST's published GR00T numbers were produced.

We **keep COAST's 12-D execution** for every arm (fixed and AAC). The AAC math does not read dims 7–11, so h* is unaffected. Only the executed rollout dynamics differ from upstream. This is recorded as an adaptation in `AAC_REPRO_NOTES.md` (A3).
