# Multi-output fixtures (QF 1.2, W0-FIX)

Made by `tests/make_multi_fixtures.py` (groups `sklearn`, `xgboost`, `ydf`, `k1`), at nice 19, one thread, 28/10/2026 (the models are seeded and trained on one thread, so the files do not depend on the machine; the K = 1 files
were compiled twice and gave the same sha256). Whole generator run: about 5 minutes. Library versions: python 3.10.13, numpy 2.2.6,
sklearn 1.7.2, xgboost 3.2.0, ydf 0.16.1.

## What is here
- sklearn (DISENO 5.5): `rf_reg_k{1,2,5,16}`, `et_reg_k3`, `dt_reg_k2`, `rf_clf_k2_c3_2` (class sets 3 and 2), `rf_clf_k1_c4`, `rf_reg_k2_nan` (trained
  with 8 % NaN; `missing_go_to_left` takes both values), each `.joblib` (test-only; QF never reads a pickle) + `.npz` with `x` (2,000 held-out rows)
  and the library's `predict` (classifiers: `proba_<k>` per output). `refused_gbr.joblib` is a GradientBoostingRegressor for the refusal test.
- xgboost: `xgb_vec_reg_k2/k5`, `xgb_vec_multilabel_k2`, `xgb_vec_softprob_c3` (all `multi_output_tree`), `xgb_mot_k1` (K = 1 under
  `multi_output_tree`), `xgb_oopt_reg_k5` (default strategy, K scalar trees per round); `.ubj` + `.npz` (`x`, `margin`, `predict` of the saved file).
- ydf: `ydf_rf_c3/` (3 classes, 300 trees, depth 8, bench data) and `ydf_rf_binary/` + `.npz` (`predict` of `model.predict`).
- **1.1 byte identity** (built from tag `v1.1` = commit `3b6324e`, a throwaway `git archive` export, never from 1.2 code):
  `k1_xgb_mot.flat`, `k1_xgb_oopt.flat` (identical: a K = 1 `multi_output_tree` model compiles to the same bytes as `one_output_per_tree`),
  `k1_ydf_rf_binary.flat`, `k1_ydf_rf_c3_replicated.flat` (1.1's K-copies compile of the 3-class RF: the bit-for-bit reference of the vector
  path); `k1_*.sha256` hold the hashes; `k1_xgb_{mot,oopt}_model.ubj` are the models they came from. Compile command: `compile.py MODEL OUT --backend flat`.
- **1.1 K = 1 prune pins**: `prune_pin_reg_raw.json`, `prune_pin_reg_log.json` (xgboost K = 1, `--regression raw|log --keep 30%`, y given) and
  `prune_pin_clf_binary_rf.json` (binary YDF RF, `--keep 30% --decision argmax`): `report.json` of 1.1's `prune.py` with timings removed.
  A K = 1 run of 1.2 must reproduce them key for key (same seed 15).

## sha256 (as built; the manifest.json has the same list)
| file | sha256 | in git |
|---|---|---|
| `dt_reg_k2.joblib` | `56623228306e25b26a337bb4a3008b2928f138cba5c4a9e5c7d49f2b79544dd1` | committed |
| `dt_reg_k2.npz` | `774c1178c376e7e56a0ce6f2afd31f0c31f523987078ae1ea7a315ead88f0ff6` | committed |
| `et_reg_k3.joblib` | `4dc0a2f6ec6a89c955444f959e926c17b1076f07a5c27dd86ec5459a3ae35b07` | regenerate (ignored by git) |
| `et_reg_k3.npz` | `7074209a8d72f9941dbadefa7755fad167bd603099ef237000ddb771faf1172f` | committed |
| `k1_xgb_mot.flat` | `373f98faca0d501a3523f099475ef39cf8a3089b1e1d639c53605e67bf36c3ad` | committed |
| `k1_xgb_mot.sha256` | `6470eb4c012b5c7a19e12107112ffe048f2bc2cd924a7f23a1e54a4065ce9fb7` | committed |
| `k1_xgb_mot_model.ubj` | `8d7b7ac8a68b722c2bf3eeabfa639116b224de3b0bd0f7da320aa60fd3911373` | committed |
| `k1_xgb_oopt.flat` | `373f98faca0d501a3523f099475ef39cf8a3089b1e1d639c53605e67bf36c3ad` | committed |
| `k1_xgb_oopt.sha256` | `6470eb4c012b5c7a19e12107112ffe048f2bc2cd924a7f23a1e54a4065ce9fb7` | committed |
| `k1_xgb_oopt_model.ubj` | `8d7b7ac8a68b722c2bf3eeabfa639116b224de3b0bd0f7da320aa60fd3911373` | committed |
| `k1_ydf_rf_binary.flat` | `cd9e4be7d747406fd73a44a1769e1f0ad3b3dbed84fc8b629c11020d556c75c2` | committed |
| `k1_ydf_rf_binary.sha256` | `dcf99fde6c3f3757076d1a18b8dbc572f537158bdb36fff2c274237a12fc93f0` | committed |
| `k1_ydf_rf_c3_replicated.flat` | `87cd2c26de19e3532532e14d4c26370357d5948877494ed92f6897ac7a719080` | committed |
| `k1_ydf_rf_c3_replicated.sha256` | `a21a77840f8286e37a0a786964be900eb7a768f8d67e6da25a5a7d5637fb2818` | committed |
| `prune_pin_clf_binary_rf.json` | `4d362c72edb7d138ef7da85a91a459790bd057978d220a5c89d2e69728fbdf80` | committed |
| `prune_pin_reg_log.json` | `4166d5b495d32aa657311d1c7a0d95b1e581af145e1eb4c3ac54ec8cb5cc0a87` | committed |
| `prune_pin_reg_raw.json` | `a1ed6beaacefcdb14a2e57310ffd9de10c05b4ed130c6aca7ee9d71243f702ff` | committed |
| `refused_gbr.joblib` | `9a45c2e1e1501f6393ac30a8627b8a099b40a8429b46f4ec97d4c99ab6ca7420` | committed |
| `rf_clf_k1_c4.joblib` | `dbd8fac11cdd3bd2b92d22614dc62aad224c224639d17ab1fadf9ecf8354bc7d` | committed |
| `rf_clf_k1_c4.npz` | `7d196795f855b5417f750252cb10698bd2b60aa232155ec23d4ba0fb1b822d21` | committed |
| `rf_clf_k2_c3_2.joblib` | `30924be588219a498d9367f8538fa00e3f00bb9fc2bc97e37cc2d616ff4b0595` | committed |
| `rf_clf_k2_c3_2.npz` | `dc659aa1148cd4df89bcbc48d4bc0a0c659c4620b0c0c4c6ca2da6f5f3153dad` | committed |
| `rf_reg_k1.joblib` | `438bb963321aa1539871edbe89b6b15bf78dc83b480d323ac0c905a51eeabb37` | committed |
| `rf_reg_k1.npz` | `f231d5d1c0029fcb589d1667b4e018ce84e55f2a42da8e02bb5b09425a956812` | committed |
| `rf_reg_k16.joblib` | `8e08882a794c30b44fa872f4a96b5df53086cb4b1e33cff2e3a9bf4a7ebdaef2` | regenerate (ignored by git) |
| `rf_reg_k16.npz` | `0f2a5caab3a7195f53be6a0a2e67d7631e3c0456ce68747a0c8665459664e68c` | committed |
| `rf_reg_k2.joblib` | `788ccf810c954d8be2d67b82a0c5d2dba02a9332fef21209d21d6bca65a59a21` | committed |
| `rf_reg_k2.npz` | `a7c7d7fd3a0ad5debbc12bc50fe37333071720492a43a2ec6618256549e234fd` | committed |
| `rf_reg_k2_nan.joblib` | `9661b8f8fb3c6c7f09a8f288c6ba94932a7e81a844e95c87c9853177908d55a2` | committed |
| `rf_reg_k2_nan.npz` | `5a3fee3f69dcb751d4d9e80537f428837b2dc7b695eac2e5c99599d384f062c2` | committed |
| `rf_reg_k5.joblib` | `b3cb801dba51629681f244793def66806b0c2b5f21cc93878b47ad7850a4f59e` | regenerate (ignored by git) |
| `rf_reg_k5.npz` | `56866b891605324b650c47fc85999adb392c20dfb52aebff0b0232ec9d56e571` | committed |
| `xgb_mot_k1.npz` | `ba13ef6c6c3eb3f1caa5373e6e69e2bc4cd8702f8128983f57c90d01b0e6f006` | committed |
| `xgb_mot_k1.ubj` | `8d7b7ac8a68b722c2bf3eeabfa639116b224de3b0bd0f7da320aa60fd3911373` | committed |
| `xgb_oopt_reg_k5.npz` | `92710d693ea42eb8a5bdd4a53b3f6bd51f4257d78ec650c416fa4f24158ff119` | committed |
| `xgb_oopt_reg_k5.ubj` | `0df193d489a0cb547d34df41e2337f29ab39d3be79f032fa4ea6aacc288c0c6c` | committed |
| `xgb_vec_multilabel_k2.npz` | `702a5b7cba0d74742d0b850b35bc75b98e32b156e3fda3198428dde2835b3c53` | committed |
| `xgb_vec_multilabel_k2.ubj` | `2301cc61bad24c4c69892960855f8199f7bb3a368cc9373296e09a9d4b418509` | committed |
| `xgb_vec_reg_k2.npz` | `6866b100e0180cd1767345a4d87df2520870c457a1458a20c0097693aebc687b` | committed |
| `xgb_vec_reg_k2.ubj` | `ce6cf89a9329a6c661ffbe89c6727a2eb14e20065dbcd25ac781781e55cbf4d9` | committed |
| `xgb_vec_reg_k5.npz` | `99a6b595a15bdaa1ab973f35229a0e512cebc356c5214d0b9ef4753e6a55955c` | committed |
| `xgb_vec_reg_k5.ubj` | `31ab6fdcdea71bd6a408241a2a4a6bfb4d1dccb490210f472708eb10024487ff` | committed |
| `xgb_vec_softprob_c3.npz` | `df962996218fa99f85ca896c0084cda47b59e302449563d5df8c164dc9d3bc87` | committed |
| `xgb_vec_softprob_c3.ubj` | `c27a224dac1a4f5ecd6e6a3a7a9261472c4c87a2e248e7ba6b5b739996023b9c` | committed |
| `ydf_rf_binary` | `722cb2b053b126d4a2ebfba4a5ed7d7c62294ff875888f9c121883bee8ce3d6f` | committed |
| `ydf_rf_binary.npz` | `08c610bcb8b7fc21fcd578fcd3b6a92cfc5a580adad527bafd62810775a31880` | committed |
| `ydf_rf_c3` | `608899c83bb2dd27363f58febd19327b0758cdfe7ad065c008260ab872980c3d` | regenerate (ignored by git) |
| `ydf_rf_c3.npz` | `f2eb5160499e6b673e98851000fdfbcc4c9f5d3f9cc778c320b73243afe4a19a` | committed |
