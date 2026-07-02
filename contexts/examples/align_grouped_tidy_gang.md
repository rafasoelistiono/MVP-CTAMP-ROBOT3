# CONTEXT.MD - Align Grouped Tidy-Up Gang Challenge

## scene
- scene_id: align_grouped_tidy_gang
- variant: align_grouped_tidy_gang

## table
- x_range: [-0.90, 0.90]
- y_range: [-0.95, 0.95]
- z_top: 0.80
- goal_center: [-0.10, 0.535, 0.833]
- goal_area_size_xy: [0.85, 0.40]

## geometry
- cube_size_xyz: [0.066, 0.066, 0.066]

## robot
- id: panda_left
- reach_min_xy: 0.30
- reach_max_xy: 0.92
- base_xy: [-0.40, 0.00]
- capabilities: [pick, place]

## objects
- id: a
  class: cube
  color: yellow
  pose: [0.10, -0.45, 0.833]
  reachable: true
  near_obstacle: false
- id: b
  class: cube
  color: green
  pose: [-0.32, -0.48, 0.833]
  reachable: true
  near_obstacle: false
- id: c
  class: cube
  color: yellow
  pose: [0.28, -0.38, 0.833]
  reachable: true
  near_obstacle: false
- id: d
  class: cube
  color: green
  pose: [-0.45, -0.62, 0.833]
  reachable: true
  near_obstacle: false
- id: e
  class: cube
  color: yellow
  pose: [0.15, -0.70, 0.833]
  reachable: true
  near_obstacle: false
- id: f
  class: cube
  color: green
  pose: [-0.32, -0.75, 0.833]
  reachable: true
  near_obstacle: false
- id: g
  class: cube
  color: blue
  pose: [0.30, -0.58, 0.833]
  reachable: true
  near_obstacle: false
- id: h
  class: cube
  color: red
  pose: [-0.62, -0.65, 0.833]
  reachable: true
  near_obstacle: false
- id: i
  class: cube
  color: blue
  pose: [0.40, -0.18, 0.833]
  reachable: true
  near_obstacle: false
- id: j
  class: cube
  color: red
  pose: [-0.68, -0.45, 0.833]
  reachable: true
  near_obstacle: false
- id: k
  class: cube
  color: blue
  pose: [0.18, -0.12, 0.833]
  reachable: true
  near_obstacle: false
- id: l
  class: cube
  color: red
  pose: [-0.55, -0.84, 0.833]
  reachable: true
  near_obstacle: false

## obstacles
- id: tall_obs_left
  pose: [-0.26, 0.35, 1.30]
  fragile: true
  radius: 0.055
  height: long
- id: tall_obs_right
  pose: [-0.12, 0.35, 1.30]
  fragile: true
  radius: 0.055
  height: long

## task
- name: align
- target_objects: [a, b, c, d, e, f, g, h, i, j, k, l]
- description: Susun 12 cube berwarna menjadi empat grup berwarna (bdf=hijau, hjl=merah, ace=kuning, gik=biru) pada goal area tanpa menggeser obstacle gang.

## constraints
- preserve_obstacles: true
- max_retries_per_object: 2
- allowed_predicates: [at, clear, holding, handempty, stable]

## grouped_tidy
- enabled: true
- require_ordered: true
- slot_prefix: tidy_slot
- axis: x
- spacing: 0.085
- row_spacing: 0.105

## tidy_groups
- id: green_top
  color: green
  objects: [b, d, f]
  center: [-0.38, 0.60, 0.833]
- id: red_bottom
  color: red
  objects: [h, j, l]
  center: [-0.38, 0.47, 0.833]
- id: yellow_top
  color: yellow
  objects: [a, c, e]
  center: [0.14, 0.60, 0.833]
- id: blue_bottom
  color: blue
  objects: [g, i, k]
  center: [0.14, 0.47, 0.833]

## challenge
- type: dual_tall_obstacle_gang
- enabled: true
- obstacle_ids: [tall_obs_left, tall_obs_right]
- require_obstacle_aware_slots: true
- require_motion_probe: true
- compare_planners: [RRTConnect, BITstar]
- min_gap_width: 0.03
- inflated_clearance_required: true
