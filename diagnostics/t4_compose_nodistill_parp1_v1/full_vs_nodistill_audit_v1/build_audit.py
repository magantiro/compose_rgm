"""Build and verify the sealed PARP1 Full-versus-NoDistill read-only audit.

This script performs no network access.  It deterministically seals the measured
facts copied from the hash-verified Modal volume audit and then verifies all
reported sums, score summaries, gaps, and lineage links.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from itertools import pairwise
from pathlib import Path
from statistics import median
from typing import Any

SCHEMA = "t4_compose_nodistill_parp1_full_vs_nodistill_audit_v1"
AUDIT_BASE_REVISION = "32f968a85de78dff81a033f6d33731ee6ceaeffe"
FULL_RUN = "b2a7e0aee53da2fa649103ac0dc83970aa00de218c90a1763f4ef8b9972397b6"
NODISTILL_RUN = "20d13545b98aab0003c9f36c128c0e9d966c0f52e01e1aefc06a863a6a18f46a"
IVG = {"0": -14.1, "1": -13.4, "2": -9.0}

# Each row is selection-input payload SHA, round-plan payload SHA, query-lock
# payload SHA, eligible A/R/S, selected S/M/L, and proposal receipt statuses.
ROUND_ROWS: dict[str, list[tuple[Any, ...]]] = {
    "full_0": [
        (
            "96cc3961763c98e608053ce6a2d35f0a70e563b80293b160a39a4a9b58790048",
            "21872d3488dc569e49bc434074e1566238dd6327ca35fa9e445eca5dddcd2329",
            "a366896f430bf313cf357001d5700e800375abf7443983938babb118bfab17e5",
            (632, 49, 558),
            (3, 4, 1),
            {"complete": 3},
        ),
        (
            "35a907ae772472edaa2791ea10364a700d8a96610fa1e94bcb25d1af3f0175c1",
            "4c6f41f6190d9e803ac3ab6ded03d1d32a033d07c92062fcad72413147396e18",
            "e535bcff2e7d5168fa90ef45a199cb4251c2c3918004599c0bcb74b8747a956d",
            (351, 80, 656),
            (4, 1, 3),
            {"complete": 12},
        ),
        (
            "78bcd859a6820952fefa13bf4e5ec7c12da23ed4f106ee34664a1fa03750feb1",
            "9c728d30021a9ea8e6a6fcf69b6279ed48659b42d27407a15adbd6e4af400d4d",
            "ec851c63d174b2df8d0f88ac96d38c171203bfa46b8ad75e285cc77f8c56678d",
            (428, 118, 876),
            (3, 5, 0),
            {"complete": 12},
        ),
        (
            "34fb972cf5db63c6f45b6fa03e59e8f462425643557b65da6b81d63bc9ebd16e",
            "c3d429e6fad9ec9674544cad54d651af50700ca59048528c34282d9052f25be3",
            "de01f1e2ac7b1967758a26c768c7ea2b766f489eb0e47eaecdf78c45425487f7",
            (41, 53, 359),
            (6, 1, 1),
            {"complete": 12},
        ),
        (
            "197e3442e6ff1fb9b79fb7c731a3b9f3c9de2f2e06ad57777ccef5587cf0f198",
            "aae9e50407c87e86e6eda166f13ef10102654cf50c6bb75e7afc731dae936914",
            "4c3f3e15bbd2c43640053cec903bd0e597ba5589da6d46f25679fe7ed931a008",
            (26, 21, 165),
            (4, 0, 4),
            {"complete": 12},
        ),
        (
            "af15e885fd5ca85a569ad8bdfd366d6bca247806c4040650d77b17061e171e0b",
            "6a02247b8c04e088f3f4300ba0830c6d0d40fc2104303e39f32e97a1b57f55d6",
            "801d9e8033313d95b86b83f2678498c911b178f44f73096d4d4926ee57900919",
            (34, 21, 128),
            (7, 0, 1),
            {"complete": 12},
        ),
    ],
    "full_1": [
        (
            "2609ddc1409fb60023a4275a61cd2fde5dc00836666cb85f19d146c4aba474bb",
            "2dc412197fe0f72c3b35a14b214a0f079d8f3546c0c3eac1d99ec399e4554fac",
            "310d6bfc4b8efd26b968a3585644fe3d987e4481b2cfd44f764e29c6cfe39c82",
            (393, 44, 346),
            (3, 4, 1),
            {"complete": 3},
        ),
        (
            "0cce4144fc6a4aead7cf386b3b8bd2bc52af3121fd3210bf4927135748ebd9a3",
            "c87f40455955a1e3f9872a84d7c2cb4f3923fec4c21e1c8b961464d8653b13c1",
            "19436df97283c92903f245fc5caa145410cd82c399e3de94b6f0cfae9f899724",
            (161, 63, 743),
            (1, 4, 3),
            {"complete": 12},
        ),
        (
            "08440601169495dfd7966b5c4e704dd034f8ec5bedad6dea343d135329d54130",
            "74181048c0566f7b29ea92ca4ad2719a0b34ae02df585e2d9345ceab0fe87763",
            "0555c7047ea713a9f165a492595ab22272e2372e86418b175a6d491f93ad7f3f",
            (6, 50, 406),
            (7, 1, 0),
            {"complete": 12},
        ),
        (
            "526be1c84796287acdda686b344b3c1e959a5f56df1652c948623dbbd819edc5",
            "a335336cb91de36c97600c6763777e317e101cd0454ff94db3a90150c22e1b54",
            "0e18e0f68d8fabd3a813db3656405fa3f02d500867355b432c79878053e2391d",
            (69, 61, 442),
            (7, 0, 1),
            {"complete": 12},
        ),
        (
            "359cd497d76c82ceef02158472d90cb970b798a9bca08584b6dd06b28d57ef6d",
            "3c383060b1ca63da33d667e00322847b416dce117e8a9393230b9afebd34eb2e",
            "a8bc3961e4e35de329ffe45ae6a5459cacff600c6ae47c0a78ccf452b684d5d6",
            (72, 67, 325),
            (8, 0, 0),
            {"complete": 12},
        ),
        (
            "cd7761a333c254d27d6a3956814a4794ffaf4c1d67d0e2743bb33866be120d31",
            "cb749931e2d0f72080f196274c4fd7476a3b4e4e5d7f1c233d9a60d1adc2e1e8",
            "286199e82573ad8c2f52ad05997c990f2b7f1a55e0a55f1a9e3cbafd7bc4e9d4",
            (65, 63, 259),
            (8, 0, 0),
            {"complete": 12},
        ),
    ],
    "full_2": [
        (
            "e41d115bbcc44b8a103b5f1fc6d92cbe8ecd1f38168aaee747489da0fa61e4ee",
            "5710d144b668862799e6a2bfd5c619bd602a5a8f08656fb230f2ff269b3c338e",
            "e06e90d01025552673779d235523584ead0476bcf59ed9657b8dcbb8b86cb4ed",
            (327, 9, 114),
            (1, 6, 1),
            {"complete": 3},
        ),
        (
            "7688b20b030a6649a32fa773d10256fb0abb279d6792cd6baa8ed82160f82211",
            "c63746e3af8b36996ce4b42b78d6fc019d5598dc2cfe46c27f7fa59987394c1a",
            "6ff648a443ce2ba9fdca8338018c2775f83c597b355b69764fd110a5b79d195d",
            (165, 32, 537),
            (4, 3, 1),
            {"complete": 11, "missing_at_deadline": 1},
        ),
        (
            "a1e1039c4850de6b5e92c35c3b359285871e88d47ea587d17a62d7f9ea3be328",
            "7ead3f5a5829b7e839b72bc47019b4f57f3426e12493060d35ba91237280fafd",
            "bd3612c847e46a006f1a86b24817f9b644f838033cef1301f9d8cdc1cf047252",
            (90, 49, 372),
            (2, 5, 1),
            {"complete": 11, "missing_at_deadline": 1},
        ),
    ],
    "nodistill_0": [
        (
            "24ae905c5f660ae2883b7bb8ca12ba3a797e97a12099a07bb14e862a3d113fd2",
            "82dcaf419b9aeeb5d306396ccd5c444b7111ff127e7d46a7eba506bacf59f912",
            "86e231922bebb187323b1aaf64e1b8b7bb340a728c4494c4f31b3fcc74e3acbd",
            (632, 6, 558),
            (4, 3, 1),
            {"complete": 3},
        ),
        (
            "f517c34f3aa937e86011dcf4e7cf337ee5f78ee4790073551c2fef94ab2a82d1",
            "68c53ffdbe3a991f4c815472b40b70195f8ae7d368bfd6b9378ef3e0aa76e1d4",
            "421b118d5b085c1a4f08dc4ad69ea5289ff06f37a03cb118850a87d804d2eb4e",
            (894, 21, 974),
            (3, 2, 3),
            {"complete": 12},
        ),
        (
            "5219312238168ae648e6d580e1611ea90f04b59335f51a7f253e506f3cf9ef5a",
            "66d734f598a895ce6c990cb44a311a6e489e45a5ae1ed077f70eb642e1ce0b7d",
            "46feeb794c0e29596d1eddb96bcd854e36f1127750930dac079f8ca190c37ec9",
            (230, 28, 634),
            (2, 6, 0),
            {"complete": 12},
        ),
        (
            "85291dd2b420354e91f15197aa4b549134bbc1383ca1e279302d63c41b5b6b66",
            "af59486d858d67f5f5a293ac8e15366c0de67e51dcbb943b29a2114fbce81c39",
            "9fd878a8ec7402c68c5e606d7545faeef5cab6343bb18c46e7ec7020ee2dc1bd",
            (63, 30, 354),
            (6, 2, 0),
            {"complete": 12},
        ),
        (
            "d7cfe895191c4cd1fe44073be40e79a09b55eb0daf8930b60bca8940e35102f1",
            "d54857ade9a6cffc6dcf9c931bb2dae89f58d69183ae2df67d11f5bc34d9a981",
            "f66a63153704c080ae020e6764cd6ac4bd4da1b74b48a6de8aa06e00c303b45b",
            (33, 56, 362),
            (5, 1, 2),
            {"complete": 12},
        ),
        (
            "331cfbd52f685d2f1f57e95548ace65186d83713af84cb81eae139f29935b61f",
            "a686bbea74521f838a479c69443b36449857774f7d75efe4c01166ce080c9133",
            "8c132d1e8c1e8621bc6996fe783aaef8266097f776ab189ef62d0a0d0a2ae434",
            (12, 44, 385),
            (6, 1, 1),
            {"complete": 12},
        ),
    ],
    "nodistill_1": [
        (
            "e5d4507200809ed818706712f32f9f86a5abea8e84148d169f218b1674d9c1ce",
            "64dbf1d5e33d4c36076fd449afe378709ef2fb655088725de91fc95247d9f409",
            "574c6060a826a56803f6edd72852e9cded46fc15899105f9f854cd8944a270b3",
            (393, 2, 346),
            (3, 4, 1),
            {"complete": 3},
        ),
        (
            "ef5354f5d61ab8e498eac40db5aebf5dfc7ccdb8ff97180e0a35eb4a19f518e6",
            "57a90ea4bc6d99651cae1a2d96bbfeb195f297526c380cb9efd08b8c29d797da",
            "bfa6ed11ebe09a2b98dc7269aaf98b03e1aeb957d9b87b293e1e030d25cda8c3",
            (94, 14, 443),
            (4, 3, 1),
            {"complete": 12},
        ),
        (
            "362c72a8049f648ed565970847ca40d27b99303765c2888f13c00a0c9d5029d1",
            "11ae1a2894b1ce8b68982c781930be5121b372aa0d1e6e29a312f5afc54ad393",
            "28449273f165a371df14d8e800acdbe3ce3ac8d5dfffee26863132d61bf60b94",
            (75, 31, 404),
            (3, 4, 1),
            {"complete": 12},
        ),
        (
            "97752c57501a6dfa79ed344bda04dcdd3d20cfc7e175916a14e68b2534464461",
            "9d14cca11d2ae2c2a5fb207b3526c85451f2028a28cc2ecd782b0b63010abedd",
            "c0f95063a2edbd8df4c5b0dc4629c7b30f58551a8261229fda8a17629456d57d",
            (21, 36, 304),
            (8, 0, 0),
            {"complete": 12},
        ),
        (
            "07db69275bf54a2c416716f693445dd4060d3737a22a8b9569f4dd20214fda2b",
            "b3b3939971bbd696cfefcf4743dc53a1060b09ffccd09c533636f3890125f339",
            "f20bef467ea2f1aae20190a667e91c6773a5b1e66f4ae4d361f804adb88df7a3",
            (20, 46, 174),
            (5, 3, 0),
            {"complete": 11, "missing_at_deadline": 1},
        ),
        (
            "29f028c055b1b63cffa823984b417c8ef1a4ffd092c8e040ffadf73b14f6e355",
            "d3aed33439d69d4c8b9b67e0dabeac242af5b693db067e5a991b9c5278c37af1",
            "a3a6534c4698b9ad71ef84c1cc984243df915801de678f74b780d908570d1902",
            (95, 39, 205),
            (6, 0, 2),
            {"complete": 12},
        ),
    ],
    "nodistill_2": [
        (
            "4debb48ebcc480e8a52b6a7f789ecc870014e408928d99b1e5691a95271d28f6",
            "b158a7f61b84a4ba23f559be9537b5389a58e127cd85257473ee4682ebbabe95",
            "84408deb2a6024296afa82b984c7db37efec47ee7fe75b790b44601029cc8a4f",
            (327, 3, 114),
            (4, 3, 1),
            {"complete": 3},
        ),
        (
            "8c09cad30510196394da86ee7d405d01a8ca7dcb0b1ccf50588fd5dc9c722f79",
            "aa605e5b3f01e60cc0128fb27a7c28e3ed33a96e0bedcb34eda896bbbc6a6a38",
            "b02e6dbee458273f2058be9ecb4ac7c0f7c5d5916cb6cd8a2b2dcf4cd52948f5",
            (413, 14, 376),
            (3, 3, 2),
            {"complete": 12},
        ),
        (
            "5a473912eaeda6d8f8357dda81782e648098ee9b11c295bc621e8c0ced17dddb",
            "d0f6bc3264a575ff60b40cff95f86ba978797a415caa2b2ebc9baeadc6ee9e82",
            "409a99e88688f905d67b3c2abff7d35c1f0343bb17fce482c4b72f17e8795225",
            (279, 13, 520),
            (2, 1, 5),
            {"complete": 12},
        ),
        (
            "4edc2c08c1cfd3ee1270165dd461e79b7e66b232223d1d66ed7726dea12731ca",
            "9f5c63707aedacf049284f47729ef3a40dff83ae2144900a8907109e9c5b86c2",
            "c72427ab53653e062b7a990093d47fdff5eb7a259aa86b9e8955e225ac4f561e",
            (67, 25, 289),
            (6, 1, 1),
            {"complete": 12},
        ),
        (
            "a58d40136a960ccf32b7570db654e0d6ffc3c2cbfd2f9942db769ce8c2ed7cad",
            "67d21464d9afbb94ee400b91d41948f3557bc7824bec1b1496c866bafe40873c",
            "b1a81a48a2fab8bcb550d9a832896281d63d70c5dc4ed4cd14a658fdb3d28d17",
            (68, 29, 278),
            (3, 1, 4),
            {"complete": 12},
        ),
        (
            "1dfcb84fbf6ce1783b9fc83295805bd1650b09e9a55b99c359cc852b1439d65f",
            "9187088a9db21a3e92441fcea3485492aa3c52c4011175dd8c981c61bb3bb566",
            "a852a5f67e485a6768021776e311330c5e13afbd9b7900b3552f39a1b35d11fb",
            (11, 27, 283),
            (5, 1, 2),
            {"complete": 12},
        ),
    ],
}

SUMMARY = {
    "full_0": {
        "checkpoint": (
            "ee3760c106f4945838a7ae87819477213eeb102467f5ae4c7df9dbbe1be2df60",
            "5a64637dea579945f7b46d181e64fd1acf085bc2620f38077a872a0406c9bd22",
        ),
        "status": "complete_budget",
        "settled_rounds": 6,
        "expert": {"route_complete_region": 13, "shallow": 25, "anchored_replacement": 12},
        "family": {
            "route_complete_region": 10,
            "substituent_delete": 18,
            "construct_substituted_ring": 12,
            "carbonyl_insert": 7,
            "bond_reroute": 4,
            "heteroatom_substitute": 4,
            "functionalize": 3,
            "fuse_ring": 2,
            "cycle_open": 2,
            "segment_replace": 2,
            "segment_grow": 2,
            "segment_shrink": 2,
            "retained_core_prune": 1,
            "append_ring": 1,
        },
        "scores": {
            "small": (27, -13.1, -11.8, -7.3, -11.193),
            "medium": (11, -12.7, -8.9, -7.8, -9.645),
            "large": (10, -12.7, -11.6, -7.5, -11.05),
        },
        "continuation": {"medium": (5, 11), "large": (2, 10)},
        "champion": -13.1,
    },
    "full_1": {
        "checkpoint": (
            "954c8101e380bbc1ec31f141d70746866fa311dba4eed22f5e1b7ed72bed1f95",
            "ff86848d3cc36d99b210ca2fa360b4312528b98201fe9bddfbd3ed7bb3966dd7",
        ),
        "status": "complete_budget",
        "settled_rounds": 6,
        "expert": {"route_complete_region": 20, "shallow": 23, "anchored_replacement": 6},
        "family": {
            "route_complete_region": 17,
            "substituent_delete": 10,
            "construct_substituted_ring": 6,
            "bond_reroute": 5,
            "functionalize": 4,
            "segment_grow": 3,
            "segment_replace": 3,
            "retained_core_prune": 3,
            "carbonyl_insert": 2,
            "segment_shrink": 2,
            "ring_system_restate": 2,
            "heteroatom_substitute": 2,
            "cycle_close": 2,
            "fuse_ring": 1,
            "append_ring": 1,
            "cycle_open": 1,
        },
        "scores": {
            "small": (34, -13.7, -12.2, -7.8, -11.803),
            "medium": (9, -10.2, -9.7, -7.0, -9.044),
            "large": (5, -13.0, -9.4, -7.2, -9.9),
        },
        "continuation": {"medium": (2, 9), "large": (4, 5)},
        "champion": -13.7,
    },
    "full_2": {
        "checkpoint": (
            "cbd768d3e0b33c87b54d0f671fb01f7cfd76463b5f601596e259f13b0bc82a57",
            "4ee62e410c278b20ca28ee359fcc43fa9040073643030166d3e48db5ad90b7c9",
        ),
        "status": "running_frozen_at_settled_round_2",
        "settled_rounds": 2,
        "expert": {"route_complete_region": 5, "shallow": 4, "anchored_replacement": 8},
        "family": {
            "route_complete_region": 4,
            "substituent_delete": 10,
            "construct_substituted_ring": 8,
            "functionalize": 1,
            "cycle_close": 1,
            "segment_shrink": 1,
            "fuse_ring": 1,
            "segment_replace": 1,
        },
        "scores": {
            "small": (5, -10.3, -8.9, -8.0, -9.02),
            "medium": (9, -11.1, -9.5, -7.1, -9.467),
            "large": (2, -10.3, -9.8, -9.3, -9.8),
        },
        "continuation": {"medium": (5, 14), "large": (1, 3)},
        "champion": -11.1,
    },
    "nodistill_0": {
        "checkpoint": (
            "260d013a89c7bd358a50c9f9708970cd54c17bcd3723b61e65198e8be3cd4bfd",
            "f72f1fd212ea1cdebf7e027c93499bce319fe391a5320faccdae25fef4808dcb",
        ),
        "status": "complete_budget",
        "settled_rounds": 6,
        "expert": {"shallow": 30, "anchored_replacement": 18, "route_complete_region": 3},
        "family": {
            "substituent_delete": 26,
            "construct_substituted_ring": 18,
            "carbonyl_insert": 9,
            "bond_reroute": 9,
            "heteroatom_substitute": 5,
            "cycle_open": 4,
            "segment_replace": 4,
            "functionalize": 3,
            "ring_system_restate": 2,
            "append_ring": 2,
            "segment_shrink": 2,
            "fuse_ring": 1,
        },
        "scores": {
            "small": (26, -10.8, -9.1, -7.3, -9.108),
            "medium": (15, -11.3, -9.1, -6.7, -9.24),
            "large": (7, -11.3, -9.3, -8.9, -9.7),
        },
        "continuation": {"medium": (6, 15), "large": (4, 7)},
        "champion": -11.3,
    },
    "nodistill_1": {
        "checkpoint": (
            "15308590402c840a0698ac23545e59ede8a9406888b072cd7478f606f394191a",
            "1b196df5e7ed150c2ebc5d505560006641625ec1e225981a89e274ec679531d8",
        ),
        "status": "complete_budget",
        "settled_rounds": 6,
        "expert": {"shallow": 37, "anchored_replacement": 11, "route_complete_region": 3},
        "family": {
            "substituent_delete": 22,
            "construct_substituted_ring": 11,
            "segment_shrink": 8,
            "bond_reroute": 7,
            "cycle_open": 6,
            "append_ring": 5,
            "functionalize": 5,
            "segment_replace": 4,
            "segment_grow": 4,
            "carbonyl_insert": 2,
            "ring_system_restate": 2,
            "fuse_ring": 1,
            "heteroatom_substitute": 1,
        },
        "scores": {
            "small": (29, -11.9, -9.8, -7.3, -9.748),
            "medium": (14, -11.6, -10.15, -8.0, -9.871),
            "large": (5, -10.1, -9.3, -8.2, -9.22),
        },
        "continuation": {"medium": (7, 14), "large": (2, 5)},
        "champion": -11.9,
    },
    "nodistill_2": {
        "checkpoint": (
            "2aa5a12e07df333d6c3def59414aa85f62a059ff139d83aae0d340c411fa5cef",
            "80129f22d9d60c38c9cd6576470ca464526ab77ca81dcae340b85878657c69b5",
        ),
        "status": "complete_budget",
        "settled_rounds": 6,
        "expert": {"shallow": 27, "anchored_replacement": 21, "route_complete_region": 4},
        "family": {
            "substituent_delete": 36,
            "construct_substituted_ring": 20,
            "bond_reroute": 5,
            "carbonyl_insert": 4,
            "segment_replace": 3,
            "fuse_ring": 3,
            "ring_system_restate": 3,
            "cycle_open": 3,
            "functionalize": 2,
            "cycle_close": 2,
            "segment_shrink": 2,
            "segment_grow": 1,
            "heteroatom_substitute": 1,
            "retained_core_prune": 1,
        },
        "scores": {
            "small": (23, -10.9, -9.7, -7.4, -9.352),
            "medium": (10, -11.6, -9.75, -7.2, -9.39),
            "large": (15, -11.4, -9.6, -7.7, -9.827),
        },
        "continuation": {"medium": (5, 10), "large": (7, 15)},
        "champion": -11.6,
    },
}

LINEAGES = json.loads(r"""{
"full_0":[[0,"parp1_0_d04_r000_q00","CN(C)Cc3ccc2c(CNC(=O)c1cccn12)c3",-7.2,null,[],"root"],[1,"parp1_0_d04_r001_q07","CN(C)Cc1ccc2c(c1)CN=C(N1CCC(O)CN1)c1cccn1-2",-8.3,"CN(C)Cc3ccc2c(CNC(=O)c1cccn12)c3",["anchored_replacement"],"exploration"],[2,"parp1_0_d04_r002_q02","O=C1NCc2cc(CCCc3ccc4c(c3)CN=C(N3CCC(F)CN3)c3cccn3-4)ccc21",-12.7,"CN(C)Cc1ccc2c(c1)CN=C(N1CCC(O)CN1)c1cccn1-2",["route_complete_region"],"route_scale_floor"],[3,"parp1_0_d04_r003_q02","O=C1NCc2cc(CCCc3ccc4c(c3)CN=C(N3CC(=O)C(F)CN3)c3cccn3-4)ccc21",-13.0,"O=C1NCc2cc(CCCc3ccc4c(c3)CN=C(N3CCC(F)CN3)c3cccn3-4)ccc21",["shallow"],"model"],[6,"parp1_0_d04_r006_q02","Cc1c(CCCc2ccc3c(c2)CN=C(N2CCCCN2)c2cccn2-3)ccc2c1CNC2=O",-13.1,"O=C1NCc2cc(CCCc3ccc4c(c3)CN=C(N3CC(=O)C(F)CN3)c3cccn3-4)ccc21",["shallow"],"model"]],
"full_1":[[0,"parp1_1_d04_r000_q00","COc1[nH]c3cccc2C(=O)NCCc1c23",-7.9,null,[],"root"],[1,"parp1_1_d04_r001_q02","CC(=O)N1CCC(c2cccc(-c3[nH]c4cccc5c4c3CCNC5=O)c2)CC1C",-13.0,"COc1[nH]c3cccc2C(=O)NCCc1c23",["route_complete_region"],"route_scale_floor"],[2,"parp1_1_d04_r002_q00","CC(=O)N1C(C)CC(c2cccc(-c3[nH]c4cccc5c4c3CCNC5=O)c2)CC1C",-13.3,"CC(=O)N1CCC(c2cccc(-c3[nH]c4cccc5c4c3CCNC5=O)c2)CC1C",["route_complete_region"],"route_scale_floor"],[3,"parp1_1_d04_r003_q06","CC(=O)N1C(C)CC(c2cccc(-c3[nH]c4cccc5c4c3CCNC5=O)c2)C(=O)C1C",-13.6,"CC(=O)N1C(C)CC(c2cccc(-c3[nH]c4cccc5c4c3CCNC5=O)c2)CC1C",["shallow"],"exploration"],[6,"parp1_1_d04_r006_q00","CC(=O)N1C(C)CC(c2cccc(-c3[nH]c4cccc5c4c3CCNC5=O)c2)C(=O)C1(C)C",-13.7,"CC(=O)N1C(C)CC(c2cccc(-c3[nH]c4cccc5c4c3CCNC5=O)c2)C(=O)C1C",["shallow"],"model"]],
"full_2":[[0,"parp1_2_d04_r000_q00","O/N=C/c1cn3CCNC(=O)c2cccc1c23",-7.5,null,[],"root"],[1,"parp1_2_d04_r001_q01","O=C1NCCn2cc(C=Nc3cccc(C(F)(F)F)c3Cl)c3cccc1c32",-10.3,"O/N=C/c1cn3CCNC(=O)c2cccc1c23",["route_complete_region"],"route_scale_floor"],[2,"parp1_2_d04_r002_q03","O=C1NCCn2cc(C=Nc3cccc(N4CCCNC4)c3Cl)c3cccc1c32",-11.1,"O=C1NCCn2cc(C=Nc3cccc(C(F)(F)F)c3Cl)c3cccc1c32",["anchored_replacement"],"expert_floor"]],
"nodistill_0":[[0,"parp1_0_d04_r000_q00","CN(C)Cc3ccc2c(CNC(=O)c1cccn12)c3",-7.5,null,[],"root"],[1,"parp1_0_d04_r001_q02","CN1CCC(c2ccc3c(c2)CNC(=O)c2cccn2-3)N1C(=O)O",-9.3,"CN(C)Cc3ccc2c(CNC(=O)c1cccn12)c3",["anchored_replacement"],"generic_scale_floor"],[3,"parp1_0_d04_r003_q04","CN1CCC(c2ccc3c(c2)CNC(=O)c2cccn2-3)N1C(=O)N1CCCN1C(=O)O",-10.3,"CN1CCC(c2ccc3c(c2)CNC(=O)c2cccn2-3)N1C(=O)O",["anchored_replacement"],"model"],[4,"parp1_0_d04_r004_q03","CN1CCCN1C(=O)N1NCCC1c1ccc2c(c1)CNC(=O)c1cccn1-2",-10.8,"CN1CCC(c2ccc3c(c2)CNC(=O)c2cccn2-3)N1C(=O)N1CCCN1C(=O)O",["shallow"],"model"],[5,"parp1_0_d04_r005_q02","CN1C(N2CCCC2)CCN1C(=O)n1nccc1-c1ccc2c(c1)CNC(=O)c1cccn1-2",-11.3,"CN1CCCN1C(=O)N1NCCC1c1ccc2c(c1)CNC(=O)c1cccn1-2",["shallow"],"model"]],
"nodistill_1":[[0,"parp1_1_d04_r000_q00","COc1[nH]c3cccc2C(=O)NCCc1c23",-8.0,null,[],"root"],[1,"parp1_1_d04_r001_q02","CC1OCCN(C(=O)O)C1c1[nH]c2cccc3c2c1CCNC3=O",-9.1,"COc1[nH]c3cccc2C(=O)NCCc1c23",["anchored_replacement"],"generic_scale_floor"],[2,"parp1_1_d04_r002_q01","CC1OCCN(C(=O)N2CCN(C(=O)O)CC2)C1c1[nH]c2cccc3c2c1CCNC3=O",-10.9,"CC1OCCN(C(=O)O)C1c1[nH]c2cccc3c2c1CCNC3=O",["anchored_replacement"],"generic_scale_floor"],[3,"parp1_1_d04_r003_q00","O=C1NCCc2c(C3C(=CF)OCCN3C(=O)N3CCN(C(=O)O)CC3)[nH]c3cccc1c23",-11.9,"CC1OCCN(C(=O)N2CCN(C(=O)O)CC2)C1c1[nH]c2cccc3c2c1CCNC3=O",["shallow"],"model"]],
"nodistill_2":[[0,"parp1_2_d04_r000_q00","O/N=C/c1cn3CCNC(=O)c2cccc1c23",-8.2,null,[],"root"],[1,"parp1_2_d04_r001_q02","CN1CCC(C(=O)O)C(c2cn3c4c(cccc24)C(=O)NCC3)N1",-9.2,"O/N=C/c1cn3CCNC(=O)c2cccc1c23",["anchored_replacement"],"generic_scale_floor"],[2,"parp1_2_d04_r002_q05","CN1CCC(N2CCCN(N)C2)=C(c2cn3c4c(cccc24)C(=O)NCC3)N1",-10.1,"CN1CCC(C(=O)O)C(c2cn3c4c(cccc24)C(=O)NCC3)N1",["anchored_replacement"],"model"],[3,"parp1_2_d04_r003_q01","CN1CCC(N2CCCN(N3CCCCN3O)C2)=C(c2cn3c4c(cccc24)C(=O)NCC3)N1",-11.6,"CN1CCC(N2CCCN(N)C2)=C(c2cn3c4c(cccc24)C(=O)NCC3)N1",["anchored_replacement"],"model"]]
}""")

UNSETTLED_LINEAGE = json.loads(
    r"""[[0,"parp1_2_d04_r000_q00","O/N=C/c1cn3CCNC(=O)c2cccc1c23",-7.5,null,[],"root"],[1,"parp1_2_d04_r001_q04","CC1CCNN(C=Cc2cn3c4c(cccc24)C(=O)NCC3)C1",-9.5,"O/N=C/c1cn3CCNC(=O)c2cccc1c23",["anchored_replacement"],"exploration"],[2,"parp1_2_d04_r002_q01","CC(=CN1CC(C)CCCN1)c1cn2c3c(cccc13)C(=O)NCC2",-10.4,"CC1CCNN(C=Cc2cn3c4c(cccc24)C(=O)NCC3)C1",["route_complete_region"],"route_scale_floor"],[3,"parp1_2_d04_r003_q03","CC(=CN1CC(N2CCN(C(=O)O)C2)=CCCN1)c1cn2c3c(cccc13)C(=O)NCC2",-11.8,"CC(=CN1CC(C)CCCN1)c1cn2c3c(cccc13)C(=O)NCC2",["anchored_replacement"],"model"]]"""
)

R3_RECEIPTS = {
    "q00": ("medium", -10.5, "8ec82554272e3d503dd34a92bb7b4907c7f60cbf9473d5779089d5074ebb1a83"),
    "q01": ("medium", -11.1, "5f162392b5f13bdcb5dc5c3f2b8bab860a4cfac6eca9effd9d13eec6fb1e2675"),
    "q02": ("medium", -9.8, "52cdecbe1efcc7db1ca1dc30b01c1a2ebf32a993f6ce16c614d763fe566730bf"),
    "q03": ("medium", -11.8, "405ccba6e19e82153b6e8f4554b92d6a045f67651d3c6a5e9210e0b80b3868d5"),
    "q04": ("small", -10.2, "48dffa23da2a5b4d4d4c95aac6dc52eceda8a8079b4691f012ebf189b0fb7800"),
    "q05": ("large", -10.7, "814b1316252ca7b880671162f5b0553bca854d678be66dd2c6ae66ef02858061"),
    "q06": ("small", None, None),
    "q07": ("medium", -10.8, "912680e3674fa016037348ab982ee58f071ce98324e1354faf080734e08b6a0f"),
}


def _lineage(rows: list[list[Any]]) -> dict[str, Any]:
    nodes = []
    for round_index, query_id, smiles, score, parent, experts, kind in rows:
        nodes.append(
            {
                "round": round_index,
                "query_id": query_id,
                "smiles": smiles,
                "score": score,
                "parent": parent,
                "proposal_experts": experts,
                "selection_kind": kind,
            }
        )
    for before, after in pairwise(nodes):
        if after["parent"] != before["smiles"]:
            raise AssertionError("champion lineage parent link drift")
    route_ancestor_rounds = [
        node["round"] for node in nodes[:-1] if "route_complete_region" in node["proposal_experts"]
    ]
    return {
        "nodes_root_to_champion": nodes,
        "final_experts": nodes[-1]["proposal_experts"],
        "has_route_derived_ancestor_before_final": bool(route_ancestor_rounds),
        "route_ancestor_rounds": route_ancestor_rounds,
    }


def _score_summary(values: tuple[Any, ...]) -> dict[str, Any]:
    count, minimum, med, maximum, mean = values
    return {"count": count, "minimum": minimum, "median": med, "maximum": maximum, "mean": mean}


def build_payload() -> dict[str, Any]:
    cells = {}
    for key, rows in ROUND_ROWS.items():
        campaign, index = key.rsplit("_", 1)
        run_id = FULL_RUN if campaign == "full" else NODISTILL_RUN
        volume = (
            f"compose-t4-shared-completion-v1-parp1-{index}-d04"
            if campaign == "full"
            else f"compose-t4-nodistill-parp1-v1-parp1-{index}-d04"
        )
        per_round = []
        for round_index, row in enumerate(rows, start=1):
            selection_sha, plan_sha, lock_sha, eligible, selected, statuses = row
            root = f"{run_id}/rounds/{round_index:03d}"
            per_round.append(
                {
                    "round": round_index,
                    "settled": round_index <= SUMMARY[key]["settled_rounds"],
                    "input_artifacts": {
                        "selection_input": {
                            "path": f"{root}/selection_input.json",
                            "payload_sha256": selection_sha,
                        },
                        "round_plan": {
                            "path": f"{root}/round_plan.json",
                            "payload_sha256": plan_sha,
                        },
                        "query_lock": {
                            "path": f"{root}/query_lock.json",
                            "payload_sha256": lock_sha,
                        },
                        "query_receipts_glob": f"{root}/query_receipts/*.json",
                    },
                    "eligible_by_expert": dict(
                        zip(
                            ("anchored_replacement", "route_complete_region", "shallow"),
                            eligible,
                            strict=True,
                        )
                    ),
                    "proposal_receipt_statuses": statuses,
                    "selected_by_band": dict(
                        zip(("small", "medium", "large"), selected, strict=True)
                    ),
                }
            )
        settled_rows = per_round[: SUMMARY[key]["settled_rounds"]]
        band_totals = {
            band: sum(row["selected_by_band"][band] for row in settled_rows)
            for band in ("small", "medium", "large")
        }
        eligible_totals = {
            expert: sum(row["eligible_by_expert"][expert] for row in per_round)
            for expert in (
                "anchored_replacement",
                "route_complete_region",
                "shallow",
            )
        }
        champion = SUMMARY[key]["champion"]
        cells[key] = {
            "campaign": campaign,
            "cell_index": int(index),
            "volume": volume,
            "run_id": run_id,
            "checkpoint": {
                "path": f"{run_id}/checkpoint.json",
                "payload_sha256": SUMMARY[key]["checkpoint"][0],
                "file_sha256": SUMMARY[key]["checkpoint"][1],
                "status": SUMMARY[key]["status"],
            },
            "rounds": per_round,
            "settled_selected_and_docked_by_band": band_totals,
            "eligible_by_expert_across_all_durable_rounds": eligible_totals,
            "settled_selected_and_docked_expert_memberships": SUMMARY[key]["expert"],
            "settled_selected_and_docked_family_token_counts": SUMMARY[key]["family"],
            "settled_docked_score_distribution_by_band": {
                band: _score_summary(values) for band, values in SUMMARY[key]["scores"].items()
            },
            "later_parent_continuation": {
                band: {"continued": values[0], "selected": values[1]}
                for band, values in SUMMARY[key]["continuation"].items()
            },
            "settled_champion_score": champion,
            "gap_to_ivg_lower_is_better": round(champion - IVG[index], 10),
            "champion_lineage": _lineage(LINEAGES[key]),
        }

    full = [cells[f"full_{index}"]["settled_champion_score"] for index in range(3)]
    nodistill = [cells[f"nodistill_{index}"]["settled_champion_score"] for index in range(3)]
    ivg = [IVG[str(index)] for index in range(3)]
    unsettled_receipts = []
    for suffix, (band, score, sha256) in R3_RECEIPTS.items():
        query_id = f"parp1_2_d04_r003_{suffix}"
        unsettled_receipts.append(
            {
                "query_id": query_id,
                "band": band,
                "score": score,
                "status": "missing" if score is None else "complete",
                "path": f"{FULL_RUN}/rounds/003/query_receipts/{query_id}.json",
                "payload_sha256": sha256,
            }
        )
    receipt_best = min(row["score"] for row in unsettled_receipts if row["score"] is not None)
    full_with_receipt = [full[0], full[1], receipt_best]
    return {
        "schema_version": SCHEMA,
        "scientific_problem": "Determine where medium and large PARP1 delta-0.4 proposals disappear in Full versus NoDistill, without changing either scored campaign.",
        "claim_boundary": "Read-only retrospective audit of sealed proposal, selection, docking, and ancestry evidence; not a new scored experiment or causal proof of trajectory distillation.",
        "audit_implementation": {
            "base_code_revision": AUDIT_BASE_REVISION,
            "script_path": "diagnostics/t4_compose_nodistill_parp1_v1/full_vs_nodistill_audit_v1/build_audit.py",
        },
        "campaign_launch_provenance": {
            "full": {
                "run_id": FULL_RUN,
                "code_revision": "54fb2d5a3f431dada67a50e6962be09d1593ee7c",
                "launch_path": f"{FULL_RUN}/launch.json",
                "launch_payload_sha256": "e2ecfb003a05c73fc47bdd20d252cee2cd2aa09b3e46aeef2b1104b659130dd4",
                "launch_file_sha256": "d88ee213023375b7036ac6951c65902510b37870d37698c8a5beade6438bf1df",
            },
            "nodistill": {
                "run_id": NODISTILL_RUN,
                "code_revision": "7fe52371f5fc049d9e81c89beed7a633ddb1d088",
                "launch_path": f"{NODISTILL_RUN}/launch.json",
                "launch_payload_sha256": "0b02ff3c791710eb7ae666e53b811a3f394c39abaf83f45d3159d395e11a29cf",
                "launch_file_sha256": "d161b6e169644122a4df8f2852a0991a1492b10a9ad5ee1be8b2d0f316441ce7",
            },
        },
        "band_definitions": {
            "nodistill": "Native proposal_scale_band on every selected row, from attach_generic_scale_band.",
            "full_route": "Native realized_primitive_band when the selected representative stores a protected primitive count.",
            "full_shallow_or_anchored": "Retrospective label, not controller-native: extent=max(1,created+deleted,regions); small<=3, medium<=11, otherwise large.",
        },
        "ivg_reference": {
            "path": "diagnostics/t4_win_audit.json",
            "file_sha256": "c07b15299a64da21c21723f17b7813dca860b24d31d316323f6e5f5a04aa5b20",
            "scores": IVG,
            "parp1_2_flagged_in_source": True,
        },
        "cells": cells,
        "full_p1_2_unsettled_round_3": {
            "checkpoint_not_advanced": True,
            "selected_by_band": {"small": 2, "medium": 5, "large": 1},
            "receipts": unsettled_receipts,
            "complete_scored_receipts": 7,
            "missing_receipts": 1,
            "missing_band": "small",
            "best_frozen_receipt_score": receipt_best,
            "gap_to_ivg_lower_is_better": round(receipt_best - IVG["2"], 10),
            "best_receipt_lineage": _lineage(UNSETTLED_LINEAGE),
        },
        "comparison": {
            "per_cell": {
                str(index): {
                    "full_settled": full[index],
                    "nodistill": nodistill[index],
                    "ivg": ivg[index],
                    "full_minus_nodistill": round(full[index] - nodistill[index], 10),
                    "full_minus_ivg": round(full[index] - ivg[index], 10),
                    "nodistill_minus_ivg": round(nodistill[index] - ivg[index], 10),
                }
                for index in range(3)
            },
            "means": {
                "full_settled": sum(full) / 3,
                "nodistill": sum(nodistill) / 3,
                "ivg": sum(ivg) / 3,
                "full_settled_minus_nodistill": sum(full) / 3 - sum(nodistill) / 3,
                "full_settled_minus_ivg": sum(full) / 3 - sum(ivg) / 3,
                "nodistill_minus_ivg": sum(nodistill) / 3 - sum(ivg) / 3,
                "full_with_unsettled_p1_2_receipt": sum(full_with_receipt) / 3,
                "full_with_unsettled_p1_2_receipt_minus_ivg": sum(full_with_receipt) / 3
                - sum(ivg) / 3,
            },
        },
        "docking_completeness": {
            "durable_selections": 264,
            "complete_scored_receipts": 263,
            "missing_receipts": 1,
            "complete_receipt_failures": 0,
            "only_missing_query_id": "parp1_2_d04_r003_q06",
            "only_missing_band": "small",
        },
        "representative_direct_proposal_receipt_census": {
            "full_0_round_1": {
                "path_prefix": f"{FULL_RUN}/rounds/001/proposal_receipts/p00_",
                "shallow_rows": 558,
                "anchored_rows": 632,
                "route_rows_by_band": {"small": 20, "medium": 10, "large": 19},
                "payload_sha256": {
                    "shallow": "059392c6f33367a19ea6990c80841b414cba95a2a7510556374a348eb847a992",
                    "anchored_replacement": "065ab34faa11b0a29afd386ef0d0490a073817277231904e0d57875d3374bb53",
                    "route_complete_region": "bf6862649cbaa870d4d7efd4b9aba1c0367bec0e69e36c3449a8a59ed8985a1d",
                },
            },
            "nodistill_0_round_1": {
                "path_prefix": f"{NODISTILL_RUN}/rounds/001/proposal_receipts/p00_",
                "shallow_rows": 558,
                "anchored_rows_by_band": {"medium": 598, "large": 34},
                "route_rows_by_band": {"small": 5, "medium": 1, "large": 0},
                "payload_sha256": {
                    "shallow": "6267863c285a170743d739a0281abdf090c77acd81fc85500db9bfb69ae199ca",
                    "anchored_replacement": "e3ac807dcfea81d86163effddbbdbd93cb27da42ef13212ac26d7c73a6c0955a",
                    "route_complete_region": "dfe9b4f013f12126e5d146765eda1179bc5618cf40a4e80cd96121386ac7b1e1",
                },
            },
        },
        "limitations": [
            "selection_input.json persists exact eligible totals only by expert, not row-level eligible candidates or eligible bands/families.",
            "Full particle-reduced route candidates are not persisted as one final row-level post-merge pool. Exact post-merge eligible-by-band counts are therefore unrecoverable.",
            "Proposal expert memberships are overlapping provenance memberships and can sum above the number of selected endpoints.",
            "Later-parent continuation counts include final-round selections, which have no later opportunity.",
            "Full p1_2 round 3 receipts are frozen but unsettled and must not be promoted into its authoritative checkpoint champion.",
        ],
        "remote_mutations": 0,
    }


def seal(payload: dict[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {"payload": payload, "payload_sha256": hashlib.sha256(encoded).hexdigest()}


def validate(payload: dict[str, Any]) -> None:
    if payload["schema_version"] != SCHEMA or payload["remote_mutations"] != 0:
        raise AssertionError("audit schema or mutation boundary drift")
    for key, cell in payload["cells"].items():
        settled = cell["rounds"][: SUMMARY[key]["settled_rounds"]]
        if any(sum(row["selected_by_band"].values()) != 8 for row in cell["rounds"]):
            raise AssertionError(f"selected batch drift in {key}")
        recomputed = {
            band: sum(row["selected_by_band"][band] for row in settled)
            for band in ("small", "medium", "large")
        }
        if recomputed != cell["settled_selected_and_docked_by_band"]:
            raise AssertionError(f"band total drift in {key}")
        if sum(recomputed.values()) != 8 * SUMMARY[key]["settled_rounds"]:
            raise AssertionError(f"settled count drift in {key}")
        if (
            cell["champion_lineage"]["nodes_root_to_champion"][-1]["score"]
            != cell["settled_champion_score"]
        ):
            raise AssertionError(f"champion lineage score drift in {key}")
        for values in SUMMARY[key]["scores"].values():
            if not (values[1] <= values[2] <= values[3]) or values[0] < 1:
                raise AssertionError(f"score summary drift in {key}")
    receipt = payload["full_p1_2_unsettled_round_3"]
    scores = [row["score"] for row in receipt["receipts"] if row["score"] is not None]
    if (
        len(scores) != 7
        or min(scores) != -11.8
        or median([row["score"] for row in receipt["receipts"] if row["band"] == "medium"]) != -10.8
    ):
        raise AssertionError("unsettled receipt summary drift")
    means = payload["comparison"]["means"]
    if (
        not math.isclose(means["full_settled"], -12.633333333333333)
        or not math.isclose(means["nodistill"], -11.6)
        or not math.isclose(means["ivg"], -12.166666666666666)
    ):
        raise AssertionError("comparison mean drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("result.json"))
    args = parser.parse_args()
    payload = build_payload()
    validate(payload)
    envelope = seal(payload)
    encoded = (json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if args.check:
        existing = args.output.read_bytes()
        if existing != encoded:
            raise SystemExit(f"determinism check failed: {args.output}")
        print(f"verified {args.output} {envelope['payload_sha256']}")
        return
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(args.output)
    print(f"wrote {args.output} {envelope['payload_sha256']}")


if __name__ == "__main__":
    main()
