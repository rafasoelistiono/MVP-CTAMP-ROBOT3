# CONTEXT.MD - Ungrouped Cubes Stack With Obstacles

## scene
- scene_id: ungroup_obs_stack_cubes
- variant: ungroup_obs

## table
- x_range: [-0.55, 0.55]
- y_range: [-0.75, 0.75]
- z_top: 0.80
- goal_center: [0.22, -0.06, 0.806]
- goal_area_size_xy: [0.52, 0.40]

## geometry
- cube_size_xyz: [0.066, 0.066, 0.066]
- cylinder_radius: 0.026
- cylinder_height: 0.08

## robot
- id: panda_left
- reach_min_xy: 0.30
- reach_max_xy: 0.82
- base_xy: [-0.4, 0.0]
- capabilities: [pick, place, stack_place]

## objects
- id: cube1
  class: cube
  pose: [-0.16, -0.42, 0.833]
  reachable: true
  near_obstacle: false
- id: circle1
  class: cylinder
  pose: [0.00, -0.54, 0.84]
  reachable: true
  near_obstacle: false
- id: cube2
  class: cube
  pose: [0.10, -0.54, 0.833]
  reachable: true
  near_obstacle: false
- id: circle2
  class: cylinder
  pose: [0.28, -0.48, 0.84]
  reachable: true
  near_obstacle: false
- id: cube3
  class: cube
  pose: [-0.10, 0.28, 0.833]
  reachable: true
  near_obstacle: false
- id: circle3
  class: cylinder
  pose: [0.06, 0.40, 0.84]
  reachable: true
  near_obstacle: false
- id: cube4
  class: cube
  pose: [0.12, 0.20, 0.833]
  reachable: true
  near_obstacle: false
- id: circle4
  class: cylinder
  pose: [0.28, 0.42, 0.84]
  reachable: true
  near_obstacle: false

## obstacles
- id: obstacle1
  pose: [0.11, -0.30, 0.89]
  fragile: true
  radius: 0.035
  height: short
- id: obstacle2
  pose: [0.35, 0.27, 0.89]
  fragile: true
  radius: 0.035
  height: short

## task
- name: stack
- target_objects: [cube1, cube2, cube3, cube4]
- description: Susun empat cube yang bercampur dengan cylinder menjadi tower cube1 sampai cube4 pada goal area tanpa menggeser obstacle.

## constraints
- preserve_obstacles: true
- max_retries_per_object: 3
- allowed_predicates: [at, on, clear, handempty, holding]

## task_plan_contract
- schema_version: ctamp-plan/v1
- output_format: Satu JSON object valid saja tanpa Markdown, code fence, komentar, atau penjelasan.
- task: stack
- slot_type: tower
- slot_axis: x
- tower_axis_note: Walaupun tower tersusun vertikal, axis wajib x karena kontrak repository. Susunan vertikal direpresentasikan oleh type tower, layer_height_m, dan on_top_of.
- geometry_rule: Hitung sendiri base_xy, base_z, dan layer_height_m dari goal, permukaan meja, serta dimensi object pada environment. Jangan menyalin angka hasil akhir dari contoh plan.
- target_rule: Gunakan seluruh target_objects dan pertahankan urutan yang dinyatakan task sebagai urutan bottom-to-top.
- step_rule: Setiap target harus memiliki tepat satu pasangan pick dan place atau stack_place. Tentukan sendiri dependency support yang konsisten dengan goal. Jangan menambahkan retry atau recovery step ke TaskPlan.
- predicate_rule: goal_predicates memakai object dengan field name dan args. Preconditions dan effects harus dihilangkan. Jika benar-benar disertakan, keduanya wajib array string seperti clear(cube1), holding(cube1), dan handempty; jangan memakai object predicate.
- constraints_rule: constraints output cukup preserve_obstacles dan bottom_to_top.

## task_plan_shape_hint

Hint berikut sengaja tidak lengkap, tidak berurutan, dan bukan JSON final yang
valid. Jangan menyalinnya secara literal. Ganti seluruh placeholder, tentukan
semua dependency dari task, lalu beri step_id integer berurutan mulai dari 0.

~~~text
{
  "schema_version": "ctamp-plan/v1",
  "task": "stack",
  "scene_id": "<scene_id dari context>",
  "target_objects": ["<urutkan semua target bottom-to-top>"],
  "goal_predicates": [
    {"name": "on", "args": ["<upper_cube>", "<lower_cube>"]},
    {"name": "at", "args": ["<base_cube>", "tower_base"]},
    "<lengkapi predicate untuk seluruh level; urutan contoh ini diacak>"
  ],
  "slot_config": {
    "type": "tower",
    "axis": "x",
    "base_xy": "<hitung dari environment>",
    "base_z": "<hitung dari environment>",
    "layer_height_m": "<hitung dari geometry object>"
  },
  "steps": [
    {"step_id": "*", "action": "stack_place", "object": "<upper_cube>", "on_top_of": "<lower_cube>"},
    {"step_id": "*", "action": "pick", "object": "<cube>"},
    {"step_id": "*", "action": "place", "object": "<base_cube>", "slot": "tower_base"},
    "<lengkapi dan urutkan semua pasangan action yang valid>"
  ],
  "constraints": {
    "preserve_obstacles": true,
    "bottom_to_top": ["<urutkan semua target>"]
  }
}
~~~

Output final wajib mengganti `*` dengan integer, mengganti semua placeholder
dengan nilai konkret dari context, memakai tipe data yang benar, dan memenuhi
urutan pick lalu place/stack_place untuk setiap cube.
