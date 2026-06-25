# CONTEXT.MD - Ungrouped Cubes Stacked Pyramid With Obstacles

## scene
- scene_id: ungroup_obs_pyramid_cubes
- variant: ungroup_obs

## table
- x_range: [-0.55, 0.55]
- y_range: [-0.75, 0.75]
- z_top: 0.80
- goal_center: [0.22, -0.06, 0.806]
- goal_area_size_xy: [0.52, 0.40]

## geometry
- cube_size_xyz: [0.066, 0.066, 0.066]

## robot
- id: panda_left
- reach_min_xy: 0.30
- reach_max_xy: 0.82
- base_xy: [-0.4, 0.0]
- capabilities: [pick, place, stack_place]

## objects
- id: cube1
  class: cube
  color: green
  pose: [-0.24, -0.48, 0.833]
  reachable: true
  near_obstacle: false
- id: cube2
  class: cube
  color: green
  pose: [-0.08, -0.58, 0.833]
  reachable: true
  near_obstacle: false
- id: cube3
  class: cube
  color: green
  pose: [0.12, -0.50, 0.833]
  reachable: true
  near_obstacle: false
- id: cube4
  class: cube
  color: red
  pose: [-0.16, 0.38, 0.833]
  reachable: true
  near_obstacle: false
- id: cube5
  class: cube
  color: red
  pose: [-0.04, 0.44, 0.833]
  reachable: true
  near_obstacle: false
- id: cube6
  class: cube
  color: yellow
  pose: [0.12, 0.38, 0.833]
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
- name: pyramid
- target_objects: [cube1, cube2, cube3, cube4, cube5, cube6]
- description: Susun enam cube menjadi stacked pyramid vertikal dengan layer 3-2-1, base layer terlebih dahulu, tanpa menggeser obstacle. Base layer berisi tiga cube hijau, layer tengah dua cube merah, dan apex satu cube kuning.

## constraints
- preserve_obstacles: true
- max_retries_per_object: 3
- allowed_predicates: [at, on, clear, handempty, holding, aligned-row]

## task_plan_contract
- schema_version: ctamp-plan/v1
- output_format: Satu JSON object valid saja tanpa Markdown, code fence, komentar, atau penjelasan.
- task: pyramid
- slot_type: pyramid
- slot_axis: x
- geometry_rule: Hitung sendiri row_count, base_row_length, spacing_m, center_x, base_y, base_z, dan layer_height_m dari goal area, permukaan meja, serta dimensi cube. row0 adalah layer bawah di permukaan meja, row1 berada di atas row0, dan row2 berada di atas row1. Semua layer centered pada center_x goal area, y tetap di base_y, z bertambah dengan layer_height_m, target aman dari obstacle buffer, dan spacing_m harus memakai lebar cube sebagai contact spacing agar support bawah rapat untuk layer atas.
- target_rule: Gunakan seluruh target_objects. Assign cube bottom-to-top; dalam setiap layer, urutkan left-to-right. Untuk 6 cube, susunan layer adalah 3, 2, 1: cube1-cube3 pada row0/layer bawah hijau, cube4-cube5 pada row1/layer tengah merah, dan cube6 pada row2/apex kuning.
- step_rule: Setiap target harus memiliki tepat satu pasangan pick lalu place ke slot row/col. Jangan menambahkan retry atau recovery step ke TaskPlan. stack_place tidak diperlukan karena slot pyramid sudah menentukan target z setiap layer.
- predicate_rule: goal_predicates memakai object dengan field name dan args. Preconditions dan effects harus dihilangkan. Jika benar-benar disertakan, keduanya wajib array string seperti clear(cube1), holding(cube1), dan handempty; jangan memakai object predicate.
- constraints_rule: constraints output cukup preserve_obstacles dan build_order.

## task_plan_shape_hint

Hint berikut sengaja tidak lengkap, tidak berurutan, dan bukan JSON final yang
valid. Jangan menyalinnya secara literal. Ganti seluruh placeholder, tentukan
slot row/col untuk semua cube, lalu beri step_id integer berurutan mulai 0.

~~~text
{
  "schema_version": "ctamp-plan/v1",
  "task": "pyramid",
  "scene_id": "<scene_id dari context>",
  "target_objects": ["<semua target dalam urutan build_order bottom-to-top>"],
  "goal_predicates": [
    {"name": "at", "args": ["<cube base kiri>", "row0_col0"]},
    {"name": "at", "args": ["<cube apex>", "row2_col0"]},
    "<lengkapi predicate at untuk semua cube>"
  ],
  "slot_config": {
    "type": "pyramid",
    "axis": "x",
    "row_count": "<hitung dari jumlah target>",
    "base_row_length": "<hitung dari jumlah target>",
    "spacing_m": "<hitung dari lebar cube agar support row bawah saling contact>",
    "center_x": "<gunakan pusat X goal area>",
    "base_y": "<gunakan pusat Y goal area yang aman dari obstacle>",
    "base_z": "<hitung dari permukaan meja dan tinggi cube>",
    "layer_height_m": "<gunakan tinggi cube untuk jarak antar-layer vertikal>"
  },
  "steps": [
    {"step_id": "*", "action": "pick", "object": "<cube>"},
    {"step_id": "*", "action": "place", "object": "<cube>", "slot": "<row_col slot>"}
  ],
  "constraints": {
    "preserve_obstacles": true,
    "build_order": ["<cube1 sampai cube6>"]
  }
}
~~~
