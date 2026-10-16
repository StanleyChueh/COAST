| task | arm_rule | arms_used | moves | simultaneous_moves | success_defines_two_arms | success_needs_both_open | success_arm_specific | eval_step_limit |
|---|---|---|---|---|---|---|---|---|
| adjust_bottle | side | 1 | 3 | 0 | False | False | False | 400 |
| beat_block_hammer | side | 1 | 3 | 0 | False | False | False | 400 |
| blocks_ranking_rgb | side | 1-2 (scene) | 5 | 1 | False | True | False | 1200 |
| blocks_ranking_size | side | 1-2 (scene) | 5 | 1 | False | True | False | 1200 |
| click_alarmclock | side | 1 | 3 | 0 | False | False | False | 400 |
| click_bell | side | 1 | 3 | 0 | False | False | False | 400 |
| dump_bin_bigbin | side+fixed | 1-2 (scene) | 8 | 1 | False | False | False | 600 |
| grab_roller | fixed-both | 2 | 2 | 2 | True | False | False | 400 |
| handover_block | side+opposite | 2 | 7 | 1 | False | False | False | 800 |
| handover_mic | side+opposite | 2 | 6 | 1 | False | False | True | 600 |
| hanging_mug | fixed-both | 2 | 8 | 1 | False | False | False | 900 |
| lift_pot | fixed-both | 2 | 3 | 3 | True | False | False | 400 |
| move_can_pot | attr | 1 | 3 | 0 | False | True | True | 400 |
| move_pillbottle_pad | side | 1 | 3 | 0 | False | True | False | 400 |
| move_playingcard_away | side | 1 | 3 | 0 | False | True | False | 400 |
| move_stapler_pad | side | 1 | 3 | 0 | False | True | False | 400 |
| open_laptop | side | 1 | 2 | 0 | False | False | True | 700 |
| open_microwave | fixed-one | 1 | 7 | 0 | False | False | False | 1500 |
| pick_diverse_bottles | fixed-both | 2 | 3 | 3 | False | False | False | 400 |
| pick_dual_bottles | fixed-both | 2 | 3 | 3 | False | False | False | 400 |
| place_a2b_left | side | 1 | 3 | 0 | False | True | False | 400 |
| place_a2b_right | side | 1 | 3 | 0 | False | True | False | 400 |
| place_bread_basket | side | 1-2 (scene) | 20 | 6 | False | True | False | 700 |
| place_bread_skillet | side+opposite | 2 | 4 | 2 | False | False | False | 500 |
| place_burger_fries | fixed-both | 2 | 6 | 3 | False | True | False | 500 |
| place_can_basket | attr+opposite | 2 | 11 | 2 | False | False | False | 700 |
| place_cans_plasticbox | fixed-both | 2 | 7 | 3 | False | True | False | 800 |
| place_container_plate | side | 1 | 4 | 0 | False | True | False | 400 |
| place_dual_shoes | fixed-both | 2 | 4 | 4 | False | True | False | 600 |
| place_empty_cup | side | 1 | 5 | 0 | False | True | False | 500 |
| place_fan | side | 1 | 3 | 0 | False | True | False | 400 |
| place_mouse_pad | side | 1 | 3 | 0 | False | True | False | 400 |
| place_object_basket | attr+opposite | 2 | 11 | 2 | False | False | False | 700 |
| place_object_scale | side | 1 | 3 | 0 | False | False | True | 400 |
| place_object_stand | side | 1 | 3 | 0 | False | True | False | 400 |
| place_phone_stand | side | 1 | 2 | 0 | False | True | False | 400 |
| place_shoe | side | 1 | 4 | 0 | False | True | False | 500 |
| press_stapler | side | 1 | 3 | 0 | False | False | False | 400 |
| put_bottles_dustbin | side+fixed | 1-2 (scene) | 10 | 2 | False | False | False | 1700 |
| put_object_cabinet | side+opposite | 2 | 5 | 0 | False | False | True | 700 |
| rotate_qrcode | side | 1 | 3 | 0 | False | True | False | 400 |
| scan_object | side+opposite | 2 | 4 | 2 | True | False | False | 500 |
| shake_bottle | side | 1 | 5 | 0 | False | False | False | 700 |
| shake_bottle_horizontally | side | 1 | 8 | 0 | False | False | False | 700 |
| stack_blocks_three | side | 1-2 (scene) | 5 | 1 | False | True | False | 1200 |
| stack_blocks_two | side | 1-2 (scene) | 5 | 1 | False | True | False | 800 |
| stack_bowls_three | side | 1-2 (scene) | 5 | 1 | False | True | False | 1200 |
| stack_bowls_two | side | 1-2 (scene) | 5 | 1 | False | True | False | 900 |
| stamp_seal | side | 1 | 3 | 0 | False | True | False | 400 |
| turn_switch | side | 1 | 2 | 0 | False | False | False | 400 |

50 tasks; arm rules: {'side': 31, 'side+fixed': 2, 'fixed-both': 8, 'side+opposite': 5, 'attr': 1, 'fixed-one': 1, 'attr+opposite': 2}
tasks with >=1 simultaneous move: 23
arms used by the expert: {'1': 26, '1-2 (scene)': 9, '2': 15}
success predicate defines two arms: ['grab_roller', 'lift_pot', 'scan_object']
success predicate is arm-specific: ['handover_mic', 'move_can_pot', 'open_laptop', 'place_object_scale', 'put_object_cabinet']
