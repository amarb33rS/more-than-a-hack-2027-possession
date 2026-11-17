# Frozen model card

```json
{
  "task": "Offline 25-Hz tracking possession, H/A/D",
  "canonical_candidate_sha256": "ef898c9ec6f380b9e80e7ddcf629bdae1d6a0695b910232695d8d8badd4fb5fb",
  "numerical_configuration": {
    "members": [
      {
        "id": "v4_42",
        "kind": "neural",
        "sha256": "6aa9de125215ff30fb2603c037dda73acc325cd0d484a5f4364961bb7f0f5005",
        "class_order": [
          0,
          1,
          2
        ],
        "architecture": "set-tcn-motion-robust-v4",
        "seed": 42,
        "epoch": 70,
        "sampling": "mixed",
        "report_sha256": "795416068d594d6cf2cf922bb5d0a85730d1d45391d6d56163bf425817ea2112",
        "train": [
          "J03WOH",
          "J03WOY",
          "J03WPY"
        ],
        "validation": [
          "J03WN1",
          "J03WQQ"
        ]
      },
      {
        "id": "v4_43",
        "kind": "neural",
        "sha256": "982173ad7a2894a7216399b0c41f5427bb8acc07b75f099b5a42c27d15842288",
        "class_order": [
          0,
          1,
          2
        ],
        "architecture": "set-tcn-motion-robust-v4",
        "seed": 43,
        "epoch": 100,
        "sampling": "mixed",
        "report_sha256": "8a6b585dc3fa9f76f573ee843c4084d188ea78c936bbfa6adfda92da33660025",
        "train": [
          "J03WOH",
          "J03WOY",
          "J03WPY"
        ],
        "validation": [
          "J03WN1",
          "J03WQQ"
        ]
      },
      {
        "id": "v4_44",
        "kind": "neural",
        "sha256": "e8c3b9cdae4ded2ca5260bf802a3e9b5917ec2c8618c67a5f10ea0ac2eaed2ec",
        "class_order": [
          0,
          1,
          2
        ],
        "architecture": "set-tcn-motion-robust-v4",
        "seed": 44,
        "epoch": 90,
        "sampling": "mixed",
        "report_sha256": "5326036a88723051d317b493b5a90b7e7e73fae13fab4381d228465e13b7e57c",
        "train": [
          "J03WOH",
          "J03WOY",
          "J03WPY"
        ],
        "validation": [
          "J03WN1",
          "J03WQQ"
        ]
      },
      {
        "id": "tree",
        "kind": "tree",
        "sha256": "86c25e35a189f2ea2c6f74c14efa04387e32b84b4536e2929a3c3beea835ee84",
        "class_order": [
          0,
          1,
          2
        ],
        "velocity_lag": 1,
        "features": [
          "ball_seen",
          "since_ball_s",
          "until_ball_s",
          "ball_x",
          "ball_y",
          "ball_speed",
          "0_distance_0",
          "0_distance_1",
          "0_distance_2",
          "0_count",
          "0_center_x",
          "0_center_y",
          "0_speed",
          "0_near_ball",
          "1_distance_0",
          "1_distance_1",
          "1_distance_2",
          "1_count",
          "1_center_x",
          "1_center_y",
          "1_speed",
          "1_near_ball",
          "ball_seen_-25",
          "since_ball_s_-25",
          "until_ball_s_-25",
          "ball_x_-25",
          "ball_y_-25",
          "ball_speed_-25",
          "0_distance_0_-25",
          "0_distance_1_-25",
          "0_distance_2_-25",
          "0_count_-25",
          "0_center_x_-25",
          "0_center_y_-25",
          "0_speed_-25",
          "0_near_ball_-25",
          "1_distance_0_-25",
          "1_distance_1_-25",
          "1_distance_2_-25",
          "1_count_-25",
          "1_center_x_-25",
          "1_center_y_-25",
          "1_speed_-25",
          "1_near_ball_-25",
          "ball_seen_25",
          "since_ball_s_25",
          "until_ball_s_25",
          "ball_x_25",
          "ball_y_25",
          "ball_speed_25",
          "0_distance_0_25",
          "0_distance_1_25",
          "0_distance_2_25",
          "0_count_25",
          "0_center_x_25",
          "0_center_y_25",
          "0_speed_25",
          "0_near_ball_25",
          "1_distance_0_25",
          "1_distance_1_25",
          "1_distance_2_25",
          "1_count_25",
          "1_center_x_25",
          "1_center_y_25",
          "1_speed_25",
          "1_near_ball_25"
        ],
        "report_sha256": "544809e99ca9d709d71bcc97ac73972123ba341d4c85e5c19360d617565673ee",
        "train": [
          "J03WOH",
          "J03WOY",
          "J03WPY"
        ],
        "validation": [
          "J03WN1",
          "J03WQQ"
        ]
      }
    ],
    "weights": {
      "v4_42": 0.25,
      "v4_43": 0.25,
      "v4_44": 0.25,
      "tree": 0.25
    },
    "switch_cost": 3,
    "train": [
      "J03WOH",
      "J03WOY",
      "J03WPY"
    ],
    "validation": [
      "J03WN1",
      "J03WQQ"
    ],
    "holdout": [
      "J03WMX",
      "J03WR9"
    ]
  },
  "independent_results": {
    "status": "completed",
    "candidate_sha256": "ef898c9ec6f380b9e80e7ddcf629bdae1d6a0695b910232695d8d8badd4fb5fb",
    "protocol_sha256": "cca463c41837ec7a2b940e4bf0dc9cd80ff8ac1c4905177d39000f80e1749a67",
    "matches": {
      "J03WMX": {
        "spell_f1_mean": 0.6416475972540046,
        "macro_f1": 0.9381946956784106,
        "alive_dead_f1": 0.9983006352108384,
        "n_gt_spells": 511,
        "n_pred_spells": 363,
        "n_graded": 145967
      },
      "J03WR9": {
        "spell_f1_mean": 0.7299999999999999,
        "macro_f1": 0.9498490991558506,
        "alive_dead_f1": 0.9964662243596495,
        "n_gt_spells": 370,
        "n_pred_spells": 310,
        "n_graded": 146810
      }
    },
    "mean": {
      "spell_f1_mean": 0.6858237986270022,
      "macro_f1": 0.9440218974171306,
      "alive_dead_f1": 0.997383429785244
    },
    "limitations": "One pre-frozen configuration on two untouched external clean matches; no alternatives, target assets or target export"
  },
  "supervision": "IDSSE provider possession labels; Bassek et al.2025, CC-BY-4.0, https://doi.org/10.1038/s41597-025-04505-y",
  "development": {
    "standard": 0.6478482725450039,
    "synthetic_noise": 0.5620120484415492,
    "target_gap_replay": 0.5946795859535146
  },
  "limitations": [
    "Only three training matches, all with Fortuna Duesseldorf at home; limited team/league diversity.",
    "Only two independent clean matches; provider proxy semantics may differ from competition control semantics.",
    "Short spells/turnovers and tracking corruption remain difficult; no target accuracy measured.",
    "Uses past/future period context and Viterbi; offline inference, not a causal real-time system.",
    "Boundary head and video proposals are not fused; no video possession improvement established.",
    "All final holdouts are consumed; further tuning requires a separate independent evaluation set."
  ]
}
```
