financial_params = {
    'symbols': ['BTC', 'ETH', 'BNB', 'SOL', 'NEAR', 'FTM', 'ADA', 'LINK', 'SHIB', 'BONK'],
    'min_collateral': 50,  # TODO check if it works
    'cooldown_period': 16, 
    'initial_balance': 11000, 
    'interval': '2h', 
    'leverage_max': 120, 
    'leverage_min': 5, 
    'limit': 450, 
    'risk_per_trade': 0.01, 
    'total_timesteps': 17500,
    'confidence_level': 0.9,
    'target_var': 0.1
}

# Define PPO hyperparameters
ppo_hyperparams = {
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'learning_rate': 3e-4,
    'clip_range': 0.2,
    'gae_lambda': 0.95,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
}



# param_1 = {'target': 2.028, 'params': {'batch_size_index': 1.251066014107722, 'clip_range': 0.31609734803264744, 'cooldown_period': 1.0021731215295528, 'ent_coef': 0.030233257263183978, 'gae_lambda': 0.8293511781634226, 'initial_balance': 5461.692973843989, 'interval': 0.9313010568883545, 'learning_rate': 0.0034621516631600474, 'leverage_max': 71.5797716499871, 'leverage_min': 11.237517946063782, 'limit': 709.5972572016474, 'max_grad_norm': 0.7796536502777316, 'n_epochs': 8.52021074275097, 'n_steps_index': 2.6343523091728365, 'risk_per_trade': 0.01520364270760597, 'total_timesteps': 36818.700407136086, 'vf_coef': 0.4755743221304143}, 'output_dir': './logs/20241010-224533'}
# param_2 = {'target': 0.688, 'params': {'batch_size_index': 3.0, 'clip_range': 0.4, 'cooldown_period': 8.153663942338756, 'ent_coef': 0.1, 'gae_lambda': 0.8293511781634226, 'initial_balance': 5465.268876376035, 'interval': 0.9313010568883545, 'learning_rate': 0.01, 'leverage_max': 75.15563861036863, 'leverage_min': 18.389008131693057, 'limit': 713.1731190804204, 'max_grad_norm': 0.7796536502777316, 'n_epochs': 15.671700928380245, 'n_steps_index': 3.0, 'risk_per_trade': 0.2, 'total_timesteps': 36822.27716826243, 'vf_coef': 1.0}, 'output_dir': './logs/20241010-225124'}
# param_3 = {'target': 0.6359999999999999, 'params': {'batch_size_index': 0.0, 'clip_range': 0.1, 'cooldown_period': 1.0, 'ent_coef': 0.0, 'gae_lambda': 0.8084324992902555, 'initial_balance': 5452.189592627246, 'interval': 1.2907660174433218, 'learning_rate': 1e-05, 'leverage_max': 20.0, 'leverage_min': 1.0, 'limit': 783.7006522234096, 'max_grad_norm': 0.9428871446522257, 'n_epochs': 3.0, 'n_steps_index': 0.0, 'risk_per_trade': 0.01, 'total_timesteps': 36726.31136087196, 'vf_coef': 0.1}, 'output_dir': './logs/20241010-225632'}
# param_4 = {'target': 0.4659999999999999, 'params': {'batch_size_index': 0.0, 'clip_range': 0.1, 'cooldown_period': 1.0, 'ent_coef': 0.0, 'gae_lambda': 0.8324365390661674, 'initial_balance': 5401.7334759485975, 'interval': 0.930180013341715, 'learning_rate': 1e-05, 'leverage_max': 20.0, 'leverage_min': 1.0, 'limit': 649.6237883830405, 'max_grad_norm': 0.7805747956397903, 'n_epochs': 3.0, 'n_steps_index': 0.0, 'risk_per_trade': 0.01, 'total_timesteps': 36758.7346666474, 'vf_coef': 0.1}, 'output_dir': './logs/20241010-225320'}
# param_5 = {'target': -12.174000000000001, 'params': {'batch_size_index': 1.6760694853372549, 'clip_range': 0.14211608157857014, 'cooldown_period': 4.763928292612697, 'ent_coef': 0.08007445686755367, 'gae_lambda': 0.9936523151438795, 'initial_balance': 6567.120890796214, 'interval': 3.4616130783465704, 'learning_rate': 0.008765127631437422, 'leverage_max': 136.29886625550017, 'leverage_min': 2.61584001602578, 'limit': 519.5273916164412, 'max_grad_norm': 0.4188812936951982, 'n_epochs': 26.709847592594155, 'n_steps_index': 0.2950405014991503, 'risk_per_trade': 0.09001044875095991, 'total_timesteps': 48315.58120602008, 'vf_coef': 0.5798487564757154}, 'output_dir': './logs/20241010-224853'}


# param_1 = {'target': 32.056, 'params': {'batch_size_index': 2.4738383797008816, 'clip_range': 0.3279427098074372, 'cooldown_period': 6.534975649758331, 'ent_coef': 0.013387994267396064, 'gae_lambda': 0.9401013357539514, 'initial_balance': 3645.6721043934836, 'interval': 4.29414368367696, 'learning_rate': 0.0013253308729928167, 'leverage_max': 40.26248280775364, 'leverage_min': 6.508665332555138, 'limit': 860.9178010755893, 'max_grad_norm': 0.9931411086480644, 'n_epochs': 19.291618914873773, 'n_steps_index': 0.1583580463573483, 'risk_per_trade': 0.03502867610157093, 'total_timesteps': 52227.80203205508, 'vf_coef': 0.579196675142013}, 'output_dir': './logs/20241011-023641'}
# param_2 = {'target': 27.160000000000004, 'params': {'batch_size_index': 1.1407871949003823, 'clip_range': 0.35902347395745393, 'cooldown_period': 19.720335135507263, 'ent_coef': 0.03226688620771313, 'gae_lambda': 0.8948005336295237, 'initial_balance': 9049.018964451281, 'interval': 4.577005145671116, 'learning_rate': 0.004078509817919743, 'leverage_max': 92.00694637030782, 'leverage_min': 3.4403037303165687, 'limit': 925.4823240317558, 'max_grad_norm': 0.9506346476571144, 'n_epochs': 12.404536181188133, 'n_steps_index': 0.5786136328067057, 'risk_per_trade': 0.12670972987610746, 'total_timesteps': 36085.38805349841, 'vf_coef': 0.542747789090202}, 'output_dir': './logs/20241011-032734'}
# param_3 = {'target': 15.602, 'params': {'batch_size_index': 1.2446012661584054, 'clip_range': 0.10761427677145428, 'cooldown_period': 18.640542626773428, 'ent_coef': 0.026428978995357844, 'gae_lambda': 0.9164094975503261, 'initial_balance': 3375.995759170997, 'interval': 2.7160727542152525, 'learning_rate': 0.008198953957209213, 'leverage_max': 93.81818749635467, 'leverage_min': 17.199984964462985, 'limit': 816.5474104961887, 'max_grad_norm': 0.4674341303319488, 'n_epochs': 4.432423157381381, 'n_steps_index': 2.4562164973634175, 'risk_per_trade': 0.06861782195930004, 'total_timesteps': 95224.37747463537, 'vf_coef': 0.5185111440535638}, 'output_dir': './logs/20241011-032924'}
# param_4 = {'target': 10.444, 'params': {'batch_size_index': 2.6791741698963167, 'clip_range': 0.380323085084469, 'cooldown_period': 12.279831129910177, 'ent_coef': 0.015918869825154735, 'gae_lambda': 0.8923395001284147, 'initial_balance': 2437.901259638839, 'interval': 3.527210510783915, 'learning_rate': 0.002959277872482559, 'leverage_max': 135.7763636943083, 'leverage_min': 12.188459472933806, 'limit': 560.7951027817771, 'max_grad_norm': 0.7834828485585676, 'n_epochs': 12.029081127947935, 'n_steps_index': 0.7747269547590141, 'risk_per_trade': 0.04310309869082702, 'total_timesteps': 90659.35993607718, 'vf_coef': 0.18531357561184514}, 'output_dir': './logs/20241011-020122'}
# param_5 = {'target': 10.218, 'params': {'batch_size_index': 2.6578262979323233, 'clip_range': 0.20718092800074994, 'cooldown_period': 18.262167867476183, 'ent_coef': 0.06233601157918028, 'gae_lambda': 0.8031642485693113, 'initial_balance': 9364.935103693851, 'interval': 3.45448458758462, 'learning_rate': 0.00997325527601029, 'leverage_max': 42.40426608489271, 'leverage_min': 3.6055792429486773, 'limit': 966.2977315185817, 'max_grad_norm': 0.7877727130429301, 'n_epochs': 4.782004663495687, 'n_steps_index': 2.2663891578073994, 'risk_per_trade': 0.15323647580763683, 'total_timesteps': 93841.96284371866, 'vf_coef': 0.7403722827656246}, 'output_dir': './logs/20241011-010909'}
# param_6 = {'target': 9.144000000000002, 'params': {'batch_size_index': 1.3980200963069245, 'clip_range': 0.22854580881634792, 'cooldown_period': 13.836055999408439, 'ent_coef': 0.09650857457482762, 'gae_lambda': 0.8432032076036001, 'initial_balance': 5860.519860990129, 'interval': 1.203785164270406, 'learning_rate': 0.004166748947721387, 'leverage_max': 78.76750354102397, 'leverage_min': 17.312006422773905, 'limit': 890.6655357795728, 'max_grad_norm': 0.774754483072926, 'n_epochs': 8.572738872815941, 'n_steps_index': 1.7340836726058733, 'risk_per_trade': 0.044063823113970495, 'total_timesteps': 69401.61672075812, 'vf_coef': 0.700896338890512}, 'output_dir': './logs/20241011-012628'}
# param_7 = {'target': 9.065999999999999, 'params': {'batch_size_index': 1.7095830003595776, 'clip_range': 0.14904041440997345, 'cooldown_period': 1.1294933192933048, 'ent_coef': 0.07361276573185996, 'gae_lambda': 0.8766864785041817, 'initial_balance': 2857.267267432596, 'interval': 4.829526819763242, 'learning_rate': 0.0029441361376642025, 'leverage_max': 34.18855063574483, 'leverage_min': 16.7155634325196, 'limit': 943.9482275092978, 'max_grad_norm': 0.6904107761114368, 'n_epochs': 7.1795597571896845, 'n_steps_index': 0.5335956891097408, 'risk_per_trade': 0.09928193253429317, 'total_timesteps': 32042.409410399363, 'vf_coef': 0.45803055209278676}, 'output_dir': './logs/20241011-021947'}
# param_8 = {'target': 8.308, 'params': {'batch_size_index': 1.3334636164331037, 'clip_range': 0.20519679671678542, 'cooldown_period': 2.4743650593519395, 'ent_coef': 0.08492505381766766, 'gae_lambda': 0.8015429370619828, 'initial_balance': 6765.5822476248695, 'interval': 1.802947960199538, 'learning_rate': 0.0017036472392340306, 'leverage_max': 63.73397389543922, 'leverage_min': 4.605173338279309, 'limit': 816.8035671075054, 'max_grad_norm': 0.8198078005452811, 'n_epochs': 20.09144999371081, 'n_steps_index': 2.5368359299070815, 'risk_per_trade': 0.19402706391634106, 'total_timesteps': 96020.31487280584, 'vf_coef': 0.420255505111252}, 'output_dir': './logs/20241011-033334'}
# param_9 = {'target': 7.7219999999999995, 'params': {'batch_size_index': 2.6969053995466874, 'clip_range': 0.16548055892715932, 'cooldown_period': 14.963763907653789, 'ent_coef': 0.06032824096146409, 'gae_lambda': 0.8554281528589486, 'initial_balance': 1157.5289857621092, 'interval': 0.6591953511137427, 'learning_rate': 0.00872309645232465, 'leverage_max': 86.4761834544486, 'leverage_min': 2.982868129271451, 'limit': 649.4180415189296, 'max_grad_norm': 0.46630904493804665, 'n_epochs': 25.070732942806675, 'n_steps_index': 2.2357951271461047, 'risk_per_trade': 0.09075602909713144, 'total_timesteps': 82560.3795503117, 'vf_coef': 0.4113438954077736}, 'output_dir': './logs/20241011-022308'}
# param_10 = {'target': 7.392000000000001, 'params': {'batch_size_index': 2.6876586545882004, 'clip_range': 0.2284273569613885, 'cooldown_period': 19.331960895819325, 'ent_coef': 0.06634414978184482, 'gae_lambda': 0.9243391440418244, 'initial_balance': 2032.7137565803766, 'interval': 4.747446293535356, 'learning_rate': 0.004504622213464606, 'leverage_max': 95.19064987032714, 'leverage_min': 8.754599252464342, 'limit': 618.5134901215139, 'max_grad_norm': 0.9323656643935776, 'n_epochs': 18.489346140151717, 'n_steps_index': 0.00861098109347691, 'risk_per_trade': 0.12725753358793754, 'total_timesteps': 46131.59214176769, 'vf_coef': 0.5743522920318483}, 'output_dir': './logs/20241011-010721'}




# params_1 = {'target': 73.43333333333334, 'params': {'batch_size_index': 0.41782904175227564, 'boost_leverage': 0.8073912887095238, 'clip_range': 0.2193030510956601, 'cooldown_period': 4.141729745221722, 'ent_coef': 0.0927508580396034, 'gae_lambda': 0.8695531719491013, 'initial_balance': 8754.060515680778, 'interval': 3.6299899267522573, 'kelly_fraction': 0.45332243648232395, 'learning_rate': 0.006240485348485533, 'leverage_max': 117.62251642355383, 'leverage_min': 7.629068497579008, 'limit': 634.963945882513, 'max_grad_norm': 0.9271203527372467, 'n_epochs': 14.558462126524962, 'n_steps_index': 2.894520141445157, 'normalize_reward': 0.6634414978184481, 'risk_per_trade': 0.12812218683973314, 'sl_atr_mult': 1.1147459729533753, 'sl_percentage': 0.09545403328363641, 'total_timesteps': 27996.48533919762, 'tp_atr_mult': 2.735168843161395, 'tp_percentage': 0.08754599252464342, 'vf_coef': 0.31332428221872494}, 'output_dir': './logs/20241020-004041'}
# params_2 = {'target': 44.166666666666664, 'params': {'batch_size_index': 1.6954501835355646, 'boost_leverage': 0.7814635845090674, 'clip_range': 0.3291729585475157, 'cooldown_period': 5.503393408354524, 'ent_coef': 0.022779545764050724, 'gae_lambda': 0.8746066133185835, 'initial_balance': 8962.107338616901, 'interval': 4.903144307072028, 'kelly_fraction': 0.10557732998020208, 'learning_rate': 0.009522194015381256, 'leverage_max': 92.34539828020476, 'leverage_min': 16.258658691746305, 'limit': 583.0057081274142, 'max_grad_norm': 0.36276305953879484, 'n_epochs': 9.499540790615592, 'n_steps_index': 0.026487871536172825, 'normalize_reward': 0.5198448324292926, 'risk_per_trade': 0.1150654331211576, 'sl_atr_mult': 1.811998911318975, 'sl_percentage': 0.09409789183772796, 'total_timesteps': 16787.474553429354, 'tp_atr_mult': 3.4647269813715234, 'tp_percentage': 0.19383996202961146, 'vf_coef': 0.1468523884509526}, 'output_dir': './logs/20241020-023655'}
# params_3 = {'target': 5.953333333333334, 'params': {'batch_size_index': 0.8633260157590462, 'boost_leverage': 0.13002857211827767, 'clip_range': 0.10581008736108913, 'cooldown_period': 13.897875125857928, 'ent_coef': 0.021162811600005906, 'gae_lambda': 0.8531093318744453, 'initial_balance': 7457.865796401691, 'interval': 0.2668127255854019, 'kelly_fraction': 0.32964704219680524, 'learning_rate': 0.0014758184633090435, 'leverage_max': 96.60971979742695, 'leverage_min': 14.295408840397693, 'limit': 551.1672144139129, 'max_grad_norm': 0.5898391914736978, 'n_epochs': 21.748804258649116, 'n_steps_index': 1.242537808580708, 'normalize_reward': 0.04995345894608716, 'risk_per_trade': 0.1118203171239472, 'sl_atr_mult': 1.663794645219789, 'sl_percentage': 0.05634002008524778, 'total_timesteps': 47783.79023963253, 'tp_atr_mult': 2.7596651215059786, 'tp_percentage': 0.18164636390469788, 'vf_coef': 0.2237272337316138}, 'output_dir': './logs/20241020-003539'}
# params_4 = {'target': 1.3933333333333333, 'params': {'batch_size_index': 0.13390187166305345, 'boost_leverage': 0.9261764372470944, 'clip_range': 0.18123572829141443, 'cooldown_period': 14.541113172835404, 'ent_coef': 0.047385308955987915, 'gae_lambda': 0.8773172414084945, 'initial_balance': 8809.484234932255, 'interval': 0.6949807724119977, 'kelly_fraction': 0.2954930891499413, 'learning_rate': 0.008413374950585417, 'leverage_max': 147.27889698739503, 'leverage_min': 2.40490028676997, 'limit': 996.0004010131552, 'max_grad_norm': 0.9680295767403924, 'n_epochs': 3.217882395075744, 'n_steps_index': 1.257144221517657, 'normalize_reward': 0.8073067824642629, 'risk_per_trade': 0.05751242428166618, 'sl_atr_mult': 1.4462744644549361, 'sl_percentage': 0.017764259451005064, 'total_timesteps': 41375.65034922248, 'tp_atr_mult': 1.4200145591605995, 'tp_percentage': 0.043531904252446095, 'vf_coef': 0.3705472906561994}, 'output_dir': './logs/20241020-004837'}
# params_5 = {'target': 1.28, 'params': {'batch_size_index': 2.7942185247043683, 'boost_leverage': 0.626481781720312, 'clip_range': 0.3819108793999588, 'cooldown_period': 7.049444320815939, 'ent_coef': 0.03869786691104976, 'gae_lambda': 0.9837971941427551, 'initial_balance': 8819.946314061215, 'interval': 3.513262817651725, 'kelly_fraction': 0.10152079051026047, 'learning_rate': 0.003118675749467038, 'leverage_max': 26.899703723515948, 'leverage_min': 6.118569883292864, 'limit': 732.3879691986481, 'max_grad_norm': 0.5788088672000797, 'n_epochs': 12.914266557886105, 'n_steps_index': 1.6624207699705003, 'normalize_reward': 0.8468752351280759, 'risk_per_trade': 0.01708101898184463, 'sl_atr_mult': 1.5357357452333273, 'sl_percentage': 0.030150021124147226, 'total_timesteps': 35158.29971644022, 'tp_atr_mult': 1.2352477753063433, 'tp_percentage': 0.028934873254926996, 'vf_coef': 0.577079165891168}, 'output_dir': './logs/20241020-024541'}
# params_6 = {'target': 0.8800000000000002, 'params': {'batch_size_index': 2.595158514680949, 'boost_leverage': 0.8448051748525726, 'clip_range': 0.37054081477265965, 'cooldown_period': 15.746253839029032, 'ent_coef': 0.08363763455350881, 'gae_lambda': 0.8055621901985976, 'initial_balance': 5687.251102211108, 'interval': 3.596805810246872, 'kelly_fraction': 0.26870498162000733, 'learning_rate': 0.0036863494180578147, 'leverage_max': 23.346858265985233, 'leverage_min': 3.4419488040455586, 'limit': 815.9448039468489, 'max_grad_norm': 0.9959286306275263, 'n_epochs': 4.846556361943334, 'n_steps_index': 1.8671338448640356, 'normalize_reward': 0.07341652443137692, 'risk_per_trade': 0.09055381915855455, 'sl_atr_mult': 1.0383132901321506, 'sl_percentage': 0.06163424252447625, 'total_timesteps': 39241.45594996286, 'tp_atr_mult': 3.230814414082508, 'tp_percentage': 0.17017627563036533, 'vf_coef': 0.8019901863496229}, 'output_dir': './logs/20241020-015252'}
# params_7 = {'target': 0.85, 'params': {'batch_size_index': 0.4720872699874292, 'boost_leverage': 0.600446505254919, 'clip_range': 0.13297206570951575, 'cooldown_period': 13.809834210269866, 'ent_coef': 0.03626761139820838, 'gae_lambda': 0.8836838035265873, 'initial_balance': 9166.608502830586, 'interval': 1.3991215474164314, 'kelly_fraction': 0.2804119289614625, 'learning_rate': 0.0021143322235376637, 'leverage_max': 126.00796763693069, 'leverage_min': 14.729944271005417, 'limit': 994.8677671326958, 'max_grad_norm': 0.4689521120726568, 'n_epochs': 17.71391201901681, 'n_steps_index': 0.3824107155296175, 'normalize_reward': 0.43035591111752947, 'risk_per_trade': 0.16241031780693715, 'sl_atr_mult': 1.3070738097766728, 'sl_percentage': 0.013834620954764607, 'total_timesteps': 31218.535342206214, 'tp_atr_mult': 2.7634433578563344, 'tp_percentage': 0.07099163870092944, 'vf_coef': 0.6244121877022126}, 'output_dir': './logs/20241020-022516'}
# params_8 = {'target': 0.6666666666666666, 'params': {'batch_size_index': 1.1040040463127276, 'boost_leverage': 0.0, 'clip_range': 0.28115300637538554, 'cooldown_period': 20.0, 'ent_coef': 0.07510472060842589, 'gae_lambda': 0.8943583154972811, 'initial_balance': 5000.0, 'interval': 0.0, 'kelly_fraction': 0.4918864227069043, 'learning_rate': 0.009381251314566337, 'leverage_max': 150.0, 'leverage_min': 1.0, 'limit': 1000.0, 'max_grad_norm': 0.3, 'n_epochs': 30.0, 'n_steps_index': 0.0, 'normalize_reward': 0.9370624202329739, 'risk_per_trade': 0.18580151430555564, 'sl_atr_mult': 2.0, 'sl_percentage': 0.08773085868176986, 'total_timesteps': 26255.771455393897, 'tp_atr_mult': 1.0, 'tp_percentage': 0.01, 'vf_coef': 1.0}, 'output_dir': './logs/20241020-005153'}
# params_9 = {'target': 0.25333333333333324, 'params': {'batch_size_index': 0.5330583204614511, 'boost_leverage': 0.5709292168543356, 'clip_range': 0.1504276774773451, 'cooldown_period': 4.056488251579704, 'ent_coef': 0.09789024308619876, 'gae_lambda': 0.8858271755526467, 'initial_balance': 5612.255938133918, 'interval': 1.211586980336425, 'kelly_fraction': 0.16556853887261905, 'learning_rate': 0.004209954631288771, 'leverage_max': 55.36399850647063, 'leverage_min': 12.414244304300803, 'limit': 963.9845797252094, 'max_grad_norm': 0.6390782202947385, 'n_epochs': 20.877858103639777, 'n_steps_index': 0.2545330320140883, 'normalize_reward': 0.09616425935626882, 'risk_per_trade': 0.08917074047736713, 'sl_atr_mult': 1.1538725414553954, 'sl_percentage': 0.07040064198159202, 'total_timesteps': 17685.249343150077, 'tp_atr_mult': 3.18904720444979, 'tp_percentage': 0.059292739490164365, 'vf_coef': 0.40637277163649876}, 'output_dir': './logs/20241020-020917'}
# params_10 = {'target': -0.1633333333333333, 'params': {'batch_size_index': 2.7101385616867613, 'boost_leverage': 0.5736794866722859, 'clip_range': 0.1008610981093477, 'cooldown_period': 12.725753358793753, 'ent_coef': 0.03266449017720962, 'gae_lambda': 0.9054116204515219, 'initial_balance': 9429.710496553873, 'interval': 1.7863488000124987, 'kelly_fraction': 0.46341406036791966, 'learning_rate': 0.006237367556760109, 'leverage_max': 22.056761570052316, 'leverage_min': 18.659307441131464, 'limit': 845.448458758462, 'max_grad_norm': 0.9981259953160364, 'n_epochs': 7.653193725323871, 'n_steps_index': 0.41140724888663327, 'normalize_reward': 0.9325954630371636, 'risk_per_trade': 0.14239545068308104, 'sl_atr_mult': 1.0660001727220625, 'sl_percentage': 0.07799167473422197, 'total_timesteps': 40155.04753844986, 'tp_atr_mult': 3.76907360663945, 'tp_percentage': 0.14518970413940965, 'vf_coef': 0.21184386577494824}, 'output_dir': './logs/20241020-004455'}


# 1.7M in training
params_1 = {'target': 37.55666666666667, 'params': {'batch_size_index': 0.3059017782773923, 'clip_range': 0.25475710509055893, 'cooldown_period': 10.065678753945864, 'ent_coef': 0.015267164409316325, 'gae_lambda': 0.9243612463480831, 'initial_balance': 58960.91069325443, 'interval': 3.2706867348537214, 'learning_rate': 0.0014540099458453517, 'leverage_max': 117.6986162275817, 'leverage_min': 5.218933656198531, 'limit': 759.6759121830165, 'max_grad_norm': 0.8497072197551332, 'n_epochs': 3.602921555778767, 'n_steps_index': 0.9730873791785595, 'risk_per_trade': 0.1758552515160211, 'total_timesteps': 87576.76860816557, 'vf_coef': 0.5845965333350893, 'boost_leverage': True, 'normalize_reward': True, 'sl_atr_mult': 1.1142648707732907, 'sl_percentage': 0.07429320973002931,'tp_atr_mult': 2.270510188551259, 'tp_percentage': 0.19871282298052018, 'kelly_fraction': 1}}
params_2 = {'target': 17.88333333333333, 'params': {'batch_size_index': 0.48023104277331097, 'clip_range': 0.2922733189322967, 'cooldown_period': 15.200985995382883, 'ent_coef': 0.04730553951678429, 'gae_lambda': 0.8319334033971203, 'initial_balance': 61707.45898048437, 'interval': 4.149733792733182, 'learning_rate': 0.00247073756741478, 'leverage_max': 101.31380040928968, 'leverage_min': 5.303092865364999, 'limit': 776.7134258107408, 'max_grad_norm': 0.5656078855576188, 'n_epochs': 23.403786878308413, 'n_steps_index': 0.03533732594198602, 'risk_per_trade': 0.19605744052467317, 'total_timesteps': 96343.8440864571, 'vf_coef': 0.8852456188919863}}
params_3 = {'target': 17.566666666666666, 'params': {'batch_size_index': 0.9961907231005109, 'clip_range': 0.1392990534432751, 'cooldown_period': 16.38032315040506, 'ent_coef': 0.034473665268329345, 'gae_lambda': 0.9880214964666734, 'initial_balance': 62381.2761952372, 'interval': 4.39415992205922, 'learning_rate': 0.008448897109468296, 'leverage_max': 137.7010014321244, 'leverage_min': 9.737725050519332, 'limit': 773.1734080101867, 'max_grad_norm': 0.8590225138064276, 'n_epochs': 10.714408996821984, 'n_steps_index': 1.470760567859783, 'risk_per_trade': 0.12383095845271688, 'total_timesteps': 21242.66204406686, 'vf_coef': 0.6341332673769735}}
params_4 = {'target': 11.910000000000002, 'params': {'batch_size_index': 2.4967146995484994, 'clip_range': 0.18984085153685581, 'cooldown_period': 12.433892659134612, 'ent_coef': 0.08915085466794334, 'gae_lambda': 0.9431723286231546, 'initial_balance': 40230.691166999866, 'interval': 3.4406972299618825, 'learning_rate': 0.004857415045119058, 'leverage_max': 80.22760091882913, 'leverage_min': 14.755846235886303, 'limit': 526.3527573444845, 'max_grad_norm': 0.626537068686245, 'n_epochs': 27.036531818231506, 'n_steps_index': 2.802514002381331, 'risk_per_trade': 0.04025908655057362, 'total_timesteps': 81691.53884708966, 'vf_coef': 0.7337484785432955}}
params_5 = {'target': 11.68, 'params': {'batch_size_index': 1.4928838550900942, 'clip_range': 0.39667045907547194, 'cooldown_period': 2.412775347021249, 'ent_coef': 0.07922896730296036, 'gae_lambda': 0.8664301498588303, 'initial_balance': 14267.611936213169, 'interval': 4.617045200223506, 'learning_rate': 0.004241499077519643, 'leverage_max': 81.84298308499783, 'leverage_min': 8.458037086821845, 'limit': 845.3917878445229, 'max_grad_norm': 0.7322677143182013, 'n_epochs': 18.67382614684403, 'n_steps_index': 2.6236052071673384, 'risk_per_trade': 0.12551939733610257, 'total_timesteps': 60874.84406209123, 'vf_coef': 0.1284522455895903}}
params_6 = {'target': 10.976666666666667, 'params': {'batch_size_index': 1.3635263397812531, 'clip_range': 0.11774700694090248, 'cooldown_period': 15.175800263032023, 'ent_coef': 0.07882844338549473, 'gae_lambda': 0.9777556475743184, 'initial_balance': 77513.6284622002, 'interval': 2.1755989605557957, 'learning_rate': 0.0036281771977551246, 'leverage_max': 142.38659805437047, 'leverage_min': 8.60304817918507, 'limit': 582.3109734453703, 'max_grad_norm': 0.45077734161059047, 'n_epochs': 20.681262117291926, 'n_steps_index': 2.0860517673498227, 'risk_per_trade': 0.15137001790519966, 'total_timesteps': 39719.78338742466, 'vf_coef': 0.9008969875261323}}
params_7 = {'target': 8.513333333333334, 'params': {'batch_size_index': 1.6383198550704112, 'clip_range': 0.21856784437134757, 'cooldown_period': 9.70399772240597, 'ent_coef': 0.04276421839872559, 'gae_lambda': 0.8188571479749411, 'initial_balance': 68058.50995268996, 'interval': 3.651935254196414, 'learning_rate': 0.007787686458866759, 'leverage_max': 121.6587334943753, 'leverage_min': 11.212751191275277, 'limit': 517.7715230931382, 'max_grad_norm': 0.7767057467325407, 'n_epochs': 6.443787011073915, 'n_steps_index': 2.503575685728185, 'risk_per_trade': 0.1737393393904072, 'total_timesteps': 85616.83229568256, 'vf_coef': 0.48070930752626473}}
params_8 = {'target': 8.093333333333334, 'params': {'batch_size_index': 1.80686218350279, 'clip_range': 0.3010934693908799, 'cooldown_period': 19.77092822074227, 'ent_coef': 0.08415757724245018, 'gae_lambda': 0.9158802279349679, 'initial_balance': 78176.44152694219, 'interval': 4.124941916048539, 'learning_rate': 0.009416108149860951, 'leverage_max': 86.1414112754312, 'leverage_min': 14.638230207820499, 'limit': 533.9479169446806, 'max_grad_norm': 0.7686874599464921, 'n_epochs': 21.783515755333642, 'n_steps_index': 2.89563646889213, 'risk_per_trade': 0.11072930871587261, 'total_timesteps': 71112.02171711257, 'vf_coef': 0.6484080657038652}}
params_9 = {'target': 8.006666666666666, 'params': {'batch_size_index': 2.83378426797244, 'clip_range': 0.2759665121505979, 'cooldown_period': 18.164636390469788, 'ent_coef': 0.013747470414623753, 'gae_lambda': 0.8278552694501518, 'initial_balance': 82665.21598385714, 'interval': 1.9883841849276678, 'learning_rate': 0.0016618884291981586, 'leverage_max': 140.5761154514844, 'leverage_min': 7.607551335164625, 'limit': 875.4060515680778, 'max_grad_norm': 0.8081985897453159, 'n_epochs': 26.849264462556864, 'n_steps_index': 1.8710166211668267, 'risk_per_trade': 0.15267906246519408, 'total_timesteps': 47911.867358227406, 'vf_coef': 0.34293510258852344}}
params_10 = {'target': 7.66, 'params': {'batch_size_index': 2.069031668964521, 'clip_range': 0.1788195911945944, 'cooldown_period': 11.446415450224114, 'ent_coef': 0.004271551634393434, 'gae_lambda': 0.8103940701029687, 'initial_balance': 92004.60353446887, 'interval': 2.360701713078659, 'learning_rate': 0.00353119691052393, 'leverage_max': 142.9741194946194, 'leverage_min': 10.692151797597992, 'limit': 732.9356236441727, 'max_grad_norm': 0.3958718996161469, 'n_epochs': 13.933612711372868, 'n_steps_index': 0.33794990843227934, 'risk_per_trade': 0.16005302858161138, 'total_timesteps': 56090.7141726554, 'vf_coef': 0.2382159126881173}}




# param_1 = {'target': 3.164, 'params': {'batch_size_index': 2.6539876468634844, 'clip_range': 0.3256635314691799, 'cooldown_period': 19.601601458300678, 'ent_coef': 0.0959006533116414, 'gae_lambda': 0.855287620968393, 'initial_balance': 5957.687498857249, 'interval': 4.199619626024045, 'kelly_fraction': 0.3629307670206786, 'learning_rate': 0.009892494723260842, 'leverage_max': 43.68647159780761, 'leverage_min': 12.781137365096924, 'limit': 648.4171881569364, 'max_grad_norm': 0.6475660694402179, 'n_epochs': 5.241642375630443, 'n_steps_index': 1.2773952601161838, 'risk_per_trade': 0.026902592512791396, 'sl_atr_mult': 1.1142648707732907, 'sl_percentage': 0.07429320973002931, 'total_timesteps': 12887.009670405436, 'tp_atr_mult': 2.270510188551259, 'tp_percentage': 0.19871282298052018, 'vf_coef': 0.8231889696761944, 'boost_leverage': True, 'normalize_reward': True}, 'output_dir': './logs/20241018-055156'}
# param_2 = {'target': 2.5300000000000002, 'params': {'batch_size_index': 0.3309278466749478, 'clip_range': 0.10825585512062111, 'cooldown_period': 17.70043864930171, 'ent_coef': 0.05163090977536197, 'gae_lambda': 0.98660463078159, 'initial_balance': 8631.065857477082, 'interval': 1.4124141638311078, 'kelly_fraction': 0.2485689434376179, 'learning_rate': 0.0011424048720281253, 'leverage_max': 92.04628395821858, 'leverage_min': 1.676964482447401, 'limit': 983.470013053018, 'max_grad_norm': 0.9921518307286485, 'n_epochs': 23.251880770664815, 'n_steps_index': 2.6059665813062654, 'risk_per_trade': 0.03561875884788194, 'sl_atr_mult': 1.4899839728423363, 'sl_percentage': 0.09167392469842016, 'total_timesteps': 23846.003133808088, 'tp_atr_mult': 1.275645986575221, 'tp_percentage': 0.08555823465787694, 'vf_coef': 0.2782196106289111}, 'output_dir': './logs/20241018-054903'}
# param_3 = {'target': -0.5900000000000001, 'params': {'batch_size_index': 1.251066014107722, 'clip_range': 0.31609734803264744, 'cooldown_period': 1.0021731215295528, 'ent_coef': 0.030233257263183978, 'gae_lambda': 0.8293511781634226, 'initial_balance': 5461.692973843989, 'interval': 0.9313010568883545, 'kelly_fraction': 0.2382242908172191, 'learning_rate': 0.003973707067564392, 'leverage_max': 90.0461754204364, 'leverage_min': 8.964695773662601, 'limit': 842.6097501983797, 'max_grad_norm': 0.4431165748120622, 'n_epochs': 26.709170782555525, 'n_steps_index': 0.08216277959377849, 'risk_per_trade': 0.13738882693389642, 'sl_atr_mult': 1.417304802367127, 'sl_percentage': 0.06028208456011766, 'total_timesteps': 15615.477543809351, 'tp_atr_mult': 1.5943044672546365, 'tp_percentage': 0.16214146804835197, 'vf_coef': 0.9714354181474578}, 'output_dir': './logs/20241018-052313'}
# param_4 = {'target': -2.408, 'params': {'batch_size_index': 0.9402725344777285, 'clip_range': 0.30769678470079426, 'cooldown_period': 17.65139389362473, 'ent_coef': 0.08946066635038474, 'gae_lambda': 0.8170088422739556, 'initial_balance': 5195.273916164412, 'interval': 0.8491520978228445, 'kelly_fraction': 0.4512570013717653, 'learning_rate': 0.0009924848699921706, 'leverage_max': 74.74399125065679, 'leverage_min': 19.199901072859536, 'limit': 766.5826424865086, 'max_grad_norm': 0.7843139797653313, 'n_epochs': 11.5189220371637, 'n_steps_index': 2.059502783044751, 'risk_per_trade': 0.16857887766050086, 'sl_atr_mult': 1.0182882773441917, 'sl_percentage': 0.07751298834504708, 'total_timesteps': 49554.44355625979, 'tp_atr_mult': 3.244496963139518, 'tp_percentage': 0.06328435849223699, 'vf_coef': 0.8103513956063396}, 'output_dir': './logs/20241018-053133'}
# param_5 = {'target': -2.8080000000000003, 'params': {'batch_size_index': 1.7237666750865688, 'clip_range': 0.39999459228581363, 'cooldown_period': 1.4748737825083997, 'ent_coef': 0.099995455619333, 'gae_lambda': 0.8293511781634226, 'initial_balance': 5461.929357093964, 'interval': 0.9313010568883545, 'kelly_fraction': 0.45266710313522135, 'learning_rate': 0.009999554241835, 'leverage_max': 90.28249550567, 'leverage_min': 9.437396392657194, 'limit': 842.8460699477828, 'max_grad_norm': 0.6794365110475432, 'n_epochs': 27.18187140155013, 'n_steps_index': 0.3184827165894473, 'risk_per_trade': 0.19999596023727628, 'sl_atr_mult': 1.8691025723636048, 'sl_percentage': 0.0999974477033408, 'total_timesteps': 15615.713965944246, 'tp_atr_mult': 2.067005128233482, 'tp_percentage': 0.19999752147771704, 'vf_coef': 0.9714354181474578}, 'output_dir': './logs/20241018-054212'}


# param_1 = {'target': 3.6719999999999997, 'params': {'batch_size_index': 2.199375960716671, 'clip_range': 0.17664055591986225, 'cooldown_period': 17.201581715371226, 'ent_coef': 0.0459568648343043, 'gae_lambda': 0.9272877952046232, 'initial_balance': 2123.3347983958915, 'interval': 2.6955710011817606, 'learning_rate': 0.007836807460442307, 'leverage_max': 42.30174690105568, 'leverage_min': 1.0, 'limit': 588.3003114195765, 'max_grad_norm': 0.47562672716955706, 'n_epochs': 3.0, 'n_steps_index': 1.6151127474083866, 'risk_per_trade': 0.05848064834562441, 'total_timesteps': 97344.3318631004, 'vf_coef': 0.3282437074491579}, 'output_dir': './logs/20240929-025631'}
# param_2 = {'target': 2.3360000000000007, 'params': {'batch_size_index': 3.0, 'clip_range': 0.1, 'cooldown_period': 20.0, 'ent_coef': 0.0, 'gae_lambda': 1.0, 'initial_balance': 2138.0589970720757, 'interval': 5.0, 'learning_rate': 0.01, 'leverage_max': 49.72024209543381, 'leverage_min': 1.0, 'limit': 600.1357112425526, 'max_grad_norm': 0.3, 'n_epochs': 3.0, 'n_steps_index': 3.0, 'risk_per_trade': 0.01, 'total_timesteps': 97339.33068410326, 'vf_coef': 0.1}, 'output_dir': './logs/20240929-024151'}
# param_3 = {'target': 1.0260000000000002, 'params': {'batch_size_index': 3.0, 'clip_range': 0.1, 'cooldown_period': 2.6843495390773913, 'ent_coef': 0.0, 'gae_lambda': 1.0, 'initial_balance': 2133.4545150505223, 'interval': 5.0, 'learning_rate': 0.01, 'leverage_max': 46.08207659867446, 'leverage_min': 1.0, 'limit': 582.0857408668105, 'max_grad_norm': 0.3, 'n_epochs': 3.0, 'n_steps_index': 3.0, 'risk_per_trade': 0.01, 'total_timesteps': 97334.39405450365, 'vf_coef': 0.1}, 'output_dir': './logs/20240929-025856'}
# param_4 = {'target': 0.638, 'params': {'batch_size_index': 0.0, 'clip_range': 0.4, 'cooldown_period': 15.405650085287864, 'ent_coef': 0.1, 'gae_lambda': 0.8, 'initial_balance': 2164.928685026995, 'interval': 0.0, 'learning_rate': 0.01, 'leverage_max': 49.90115243232055, 'leverage_min': 1.0, 'limit': 586.1018631222252, 'max_grad_norm': 1.0, 'n_epochs': 3.0, 'n_steps_index': 0.0, 'risk_per_trade': 0.2, 'total_timesteps': 97324.08915900251, 'vf_coef': 1.0}, 'output_dir': './logs/20240929-023200'}
# param_5 = {'target': 0.49800000000000005, 'params': {'batch_size_index': 0.0, 'clip_range': 0.4, 'cooldown_period': 1.0, 'ent_coef': 0.0, 'gae_lambda': 0.8, 'initial_balance': 2169.581347574117, 'interval': 0.0, 'learning_rate': 0.01, 'leverage_max': 20.0, 'leverage_min': 1.0, 'limit': 579.9425437944539, 'max_grad_norm': 1.0, 'n_epochs': 3.0, 'n_steps_index': 0.0, 'risk_per_trade': 0.2, 'total_timesteps': 97324.02377904895, 'vf_coef': 1.0}, 'output_dir': './logs/20240929-020009'}
# param_6 = {'target': 0.41600000000000004, 'params': {'batch_size_index': 0.0, 'clip_range': 0.4, 'cooldown_period': 4.378064389715177, 'ent_coef': 0.1, 'gae_lambda': 0.8, 'initial_balance': 2123.0356607943013, 'interval': 0.0, 'learning_rate': 0.004741994767598638, 'leverage_max': 55.314995629196325, 'leverage_min': 1.0, 'limit': 607.6848523283118, 'max_grad_norm': 1.0, 'n_epochs': 3.0, 'n_steps_index': 0.0, 'risk_per_trade': 0.2, 'total_timesteps': 97337.67780987022, 'vf_coef': 1.0}, 'output_dir': './logs/20240929-024542'}
# param_7 = {'target': 0.356, 'params': {'batch_size_index': 3.0, 'clip_range': 0.1, 'cooldown_period': 1.0, 'ent_coef': 0.0, 'gae_lambda': 0.8592753180290698, 'initial_balance': 2151.8160853839245, 'interval': 0.0, 'learning_rate': 1e-05, 'leverage_max': 36.27147206006032, 'leverage_min': 1.0, 'limit': 609.3006111905536, 'max_grad_norm': 0.3, 'n_epochs': 3.0, 'n_steps_index': 0.0, 'risk_per_trade': 0.01, 'total_timesteps': 97315.4700148064, 'vf_coef': 0.1}, 'output_dir': './logs/20240929-023359'}
# param_8 = {'target': 0.27599999999999997, 'params': {'batch_size_index': 0.0, 'clip_range': 0.1, 'cooldown_period': 2.0560230869374285, 'ent_coef': 0.0, 'gae_lambda': 0.9125429531441807, 'initial_balance': 2211.990774082957, 'interval': 0.0, 'learning_rate': 1e-05, 'leverage_max': 20.0, 'leverage_min': 1.0, 'limit': 596.7539420537098, 'max_grad_norm': 0.7994509791946871, 'n_epochs': 3.0, 'n_steps_index': 0.0, 'risk_per_trade': 0.01, 'total_timesteps': 97366.00334711897, 'vf_coef': 0.6125749296726888}, 'output_dir': './logs/20240929-002657'}
# param_9 = {'target': 0.21800000000000003, 'params': {'batch_size_index': 0.0, 'clip_range': 0.4, 'cooldown_period': 7.802243738746445, 'ent_coef': 0.1, 'gae_lambda': 0.9640599078383864, 'initial_balance': 2236.4137719218597, 'interval': 0.0, 'learning_rate': 0.01, 'leverage_max': 20.0, 'leverage_min': 1.0, 'limit': 611.7796788054281, 'max_grad_norm': 0.9094999089860953, 'n_epochs': 6.838108290274632, 'n_steps_index': 0.0, 'risk_per_trade': 0.2, 'total_timesteps': 97361.57171223265, 'vf_coef': 0.8698208952207632}, 'output_dir': './logs/20240929-010657'}
# param_10 = {'target': 0.188, 'params': {'batch_size_index': 0.0, 'clip_range': 0.4, 'cooldown_period': 20.0, 'ent_coef': 0.1, 'gae_lambda': 0.8, 'initial_balance': 2122.9953629264137, 'interval': 0.0, 'learning_rate': 0.01, 'leverage_max': 52.98652135944634, 'leverage_min': 1.0, 'limit': 595.6265361283922, 'max_grad_norm': 1.0, 'n_epochs': 3.0, 'n_steps_index': 0.0, 'risk_per_trade': 0.2, 'total_timesteps': 97347.15088823326, 'vf_coef': 1.0}, 'output_dir': './logs/20240929-031536'}





# params_1 = {'target': 3129.78, 'params': {'batch_size_index': 1.5844211179507535, 'clip_range': 0.13237127266215884, 'cooldown_period': 9.354994239820977, 'ent_coef': 0.07031109514274603, 'gae_lambda': 0.8894069126871217, 'initial_balance': 5560.946829219889, 'interval': 10.339085442294866, 'learning_rate': 0.002913367059244244, 'leverage_max': 131.40980675394405, 'leverage_min': 5.821668426040751, 'limit': 958.053811402179, 'max_grad_norm': 0.3354230325442903, 'n_epochs': 13.22131944507182, 'n_steps_index': 1.2432596600652799, 'risk_per_trade': 0.027468255908579993, 'total_timesteps': 11734.87458556929, 'vf_coef': 0.8348355943377369}}
# params_2 = {'target': 136.61, 'params': {'batch_size_index': 0.6243214969592026, 'clip_range': 0.1287444669337641, 'cooldown_period': 11.672446961373076, 'ent_coef': 0.09994179938969672, 'gae_lambda': 0.9563585894805005, 'initial_balance': 5897.202072338772, 'interval': 8.23415449138731, 'learning_rate': 0.0006428414690001771, 'leverage_max': 88.37831594251504, 'leverage_min': 12.586990903136847, 'limit': 941.7017957946985, 'max_grad_norm': 0.5332958397465413, 'n_epochs': 24.69187909943415, 'n_steps_index': 0.6055513221614522, 'risk_per_trade': 0.1426369881507166, 'total_timesteps': 19580.002627868802, 'vf_coef': 0.2552545931295582}}
# params_3 = {'target': 99.72, 'params': {'batch_size_index': 1.828315042424443, 'clip_range': 0.1734171235373495, 'cooldown_period': 3.800998964199468, 'ent_coef': 0.06344939427924867, 'gae_lambda': 0.9802435607357757, 'initial_balance': 8912.442394425354, 'interval': 8.345675214074252, 'learning_rate': 0.0007893270550499326, 'leverage_max': 149.04806085508534, 'leverage_min': 10.196782329474555, 'limit': 819.0914815541767, 'max_grad_norm': 0.8048188480010146, 'n_epochs': 14.675636112668572, 'n_steps_index': 2.547463199049475, 'risk_per_trade': 0.041719288714865424, 'total_timesteps': 36153.1525552246, 'vf_coef': 0.5174707515372764}}
# params_4 = {'target': 92.16, 'params': {'batch_size_index': 0.24155469564461418, 'clip_range': 0.33295715931857817, 'cooldown_period': 8.338409448144427, 'ent_coef': 0.08917694208706466, 'gae_lambda': 0.8972841829962688, 'initial_balance': 9570.734990340276, 'interval': 9.548736624507455, 'learning_rate': 0.0008454488275744148, 'leverage_max': 82.32975558521451, 'leverage_min': 18.231054422199772, 'limit': 839.4747354981796, 'max_grad_norm': 0.7389546536918166, 'n_epochs': 4.448958380699626, 'n_steps_index': 1.9936783414146122, 'risk_per_trade': 0.1790549713078917, 'total_timesteps': 23960.073542674043, 'vf_coef': 0.5757637288037674}}
# params_5 = {'target': 64.55, 'params': {'batch_size_index': 0.4410127738914993, 'clip_range': 0.3692746913886712, 'cooldown_period': 15.023822047344954, 'ent_coef': 0.08470758039353177, 'gae_lambda': 0.9404880401986586, 'initial_balance': 6712.56201659857, 'interval': 8.613118994423255, 'learning_rate': 0.003185236437567369, 'leverage_max': 127.10183117011323, 'leverage_min': 11.971168449949955, 'limit': 945.9625731446254, 'max_grad_norm': 0.5070949961671266, 'n_epochs': 25.896089745986735, 'n_steps_index': 2.0687640809448, 'risk_per_trade': 0.18793077934783053, 'total_timesteps': 28545.56612468381, 'vf_coef': 0.34592569165051845}}
# params_6 = {'target': 52.07, 'params': {'batch_size_index': 3.0, 'clip_range': 0.17130133787934307, 'cooldown_period': 20.0, 'ent_coef': 0.1, 'gae_lambda': 0.8, 'initial_balance': 9544.102509614517, 'interval': 10.983104028479318, 'learning_rate': 0.01, 'leverage_max': 20.0, 'leverage_min': 20.0, 'limit': 1000.0, 'max_grad_norm': 1.0, 'n_epochs': 3.0, 'n_steps_index': 0.16660390473562556, 'risk_per_trade': 0.2, 'total_timesteps': 48831.95323186855, 'vf_coef': 0.5443369487445826}}
# params_7 = {'target': 32.62, 'params': {'batch_size_index': 2.7471225564953334, 'clip_range': 0.16396813551763345, 'cooldown_period': 11.334510557328253, 'ent_coef': 0.04253123916936155, 'gae_lambda': 0.8295572600669375, 'initial_balance': 7056.654705918428, 'interval': 8.255926662647704, 'learning_rate': 0.0060520224200817235, 'leverage_max': 70.15250945210984, 'leverage_min': 11.844678290869805, 'limit': 686.9430275530697, 'max_grad_norm': 0.8145553793622238, 'n_epochs': 3.071476782543035, 'n_steps_index': 2.864665944785925, 'risk_per_trade': 0.18018587392433208, 'total_timesteps': 35390.63744926753, 'vf_coef': 0.15482520310976888}}
# params_8 = {'target': 28.2, 'params': {'batch_size_index': 3.0, 'clip_range': 0.17130133787934307, 'cooldown_period': 20.0, 'ent_coef': 0.1, 'gae_lambda': 0.8, 'initial_balance': 9488.080955168096, 'interval': 9.948942919587457, 'learning_rate': 0.01, 'leverage_max': 20.0, 'leverage_min': 20.0, 'limit': 1000.0, 'max_grad_norm': 1.0, 'n_epochs': 3.0, 'n_steps_index': 0.16660390473562556, 'risk_per_trade': 0.2, 'total_timesteps': 48831.95323186855, 'vf_coef': 0.5443369487445826}}
# params_9 = {'target': 23.45, 'params': {'batch_size_index': 2.6578262979323233, 'clip_range': 0.20718092800074994, 'cooldown_period': 18.262167867476183, 'ent_coef': 0.06233601157918028, 'gae_lambda': 0.8031642485693113, 'initial_balance': 9647.186168718807, 'interval': 7.599866092686163, 'learning_rate': 0.00997325527601029, 'leverage_max': 42.40426608489271, 'leverage_min': 3.6055792429486773, 'limit': 966.2977315185817, 'max_grad_norm': 0.7877727130429301, 'n_epochs': 4.782004663495687, 'n_steps_index': 2.2663891578073994, 'risk_per_trade': 0.15323647580763683, 'total_timesteps': 46920.98142185933, 'vf_coef': 0.7403722827656246}}
# params_10 =  {'target': 19.1, 'params': {'batch_size_index': 2.414246255241734, 'clip_range': 0.17658519959770744, 'cooldown_period': 10.617658879661159, 'ent_coef': 0.035915464808904576, 'gae_lambda': 0.8193856198706664, 'initial_balance': 8505.23181372088, 'interval': 4.100259892171297, 'learning_rate': 0.001443130639165824, 'leverage_max': 39.45806918172654, 'leverage_min': 11.511182384144902, 'limit': 900.1370444999577, 'max_grad_norm': 0.9432106919102001, 'n_epochs': 7.524351749367117, 'n_steps_index': 2.568119428896631, 'risk_per_trade': 0.03244537883520713, 'total_timesteps': 37030.20524880984, 'vf_coef': 0.8917331139938193}}

n_steps_list = [512, 1024, 2048, 4096]
batch_sizes = [32, 64, 128, 256]
possible_intervals = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']

# params1 = {'target': 54.34, 'params': {'cooldown_period': 11.321819633343276, 'initial_balance': 19871.069924491472, 'interval': 4.366778811643908, 'leverage_max': 132.41342769054663, 'leverage_min': 14.035442866121741, 'limit': 979.7789064449515, 'risk_per_trade': 0.05407951196573209, 'total_timesteps': 39411.238609576496}}
# params2 = {'target': 34.11, 'params': {'cooldown_period': 11.733533355502718, 'initial_balance': 22743.51289728541, 'interval': 5.228633496836138, 'leverage_max': 50.1293549508441, 'leverage_min': 8.118834144318448, 'limit': 413.9055636952672, 'risk_per_trade': 0.25, 'total_timesteps': 27807.575872764894}}
# params3 = {'target': 24.05, 'params': {'cooldown_period': 16.133497140466723, 'initial_balance': 23878.713781093604, 'interval': 6.471178049207773, 'leverage_max': 64.09585507025832, 'leverage_min': 6.22298476704464, 'limit': 448.5423535633668, 'risk_per_trade': 0.012187497377328809, 'total_timesteps': 26699.701458320284}}
# params4 = {'target': 22.25, 'params': {'cooldown_period': 14.701103293927659, 'initial_balance': 21755.368772224534, 'interval': 7.887067966962987, 'leverage_max': 71.98669767085607, 'leverage_min': 7.821665097406508, 'limit': 396.51126442727264, 'risk_per_trade': 0.1764089025067637, 'total_timesteps': 78038.84737917819}}
# params5 = {'target': 19.07, 'params': {'cooldown_period': 8.050637720476072, 'initial_balance': 16474.846958223494, 'interval': 2.6799062958270614, 'leverage_max': 102.15012268259142, 'leverage_min': 19.27330917518158, 'limit': 643.0903282111348, 'risk_per_trade': 0.11148101492021657, 'total_timesteps': 48727.4433203517}}

from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

constant_params = {
    'symbols': ['BTC', 'ETH', 'BNB', 'SOL', 'NEAR', 'FTM', 'ADA', 'LINK', 'SHIB', 'BONK'],
    'min_collateral': 50,  # TODO check if it works
    'period_start': os.getenv('PERIOD_START'),
    'period_end': os.getenv('PERIOD_END'),
    'data_split': float(os.getenv('DATA_SPLIT')),
}

def create_financial_params(params):
    def round_value(key, value):
        if isinstance(value, (int, float)):
            if key.startswith(('sl', 'tp')):
                return round(value, 2)
            elif value < 1:
                return round(value, 2)
            else:
                return round(value)
        return value

    financial_params = {key: round_value(key, value) for key, value in params['params'].items() if key not in [
        'n_steps_index', 'batch_size_index', 'n_epochs', 'learning_rate', 'clip_range', 'gae_lambda', 'ent_coef', 'vf_coef', 'max_grad_norm']}
    
    # Ensure boost_leverage and normalize_reward are boolean
    financial_params['boost_leverage'] = bool(params['params'].get('boost_leverage', False))
    financial_params['normalize_reward'] = bool(params['params'].get('normalize_reward', False))
    
    # Map interval index to actual interval value
    interval_index = financial_params.get('interval', 0)
    if isinstance(interval_index, (int, float)):
        if 0 <= interval_index < len(possible_intervals):
            financial_params['interval'] = possible_intervals[int(interval_index)]
        else:
            raise ValueError('Interval was not found. Exiting training...')
    else:
        financial_params['interval'] = interval_index

    financial_params['symbols'] = constant_params['symbols']
    financial_params['min_collateral'] = constant_params['min_collateral']
    financial_params['period_start'] = constant_params['period_start']
    financial_params['period_end'] = constant_params['period_end']
    financial_params['data_split'] = constant_params['data_split']
    # financial_params['confidence_level'] = 0.9
    # financial_params['target_var'] = 0.1
    return financial_params


def create_ppo_hyperparams(params):
    def round_hyperparam(value):
        if isinstance(value, (int, float)):
            if value < 1:
                return round(value, 3)
            else:
                return value
        return value

    interval_index = params['params'].get('interval', 0)
    if not isinstance(interval_index, (int, float)):
        return {
            'n_steps': params['params']['n_steps'],
            'batch_size': params['params']['batch_size'],
            'n_epochs': int(params['params']['n_epochs']),
            'learning_rate': round_hyperparam(params['params']['learning_rate']),
            'clip_range': round_hyperparam(params['params']['clip_range']),
            'gae_lambda': round_hyperparam(params['params']['gae_lambda']),
            'ent_coef': round_hyperparam(params['params']['ent_coef']),
            'vf_coef': round_hyperparam(params['params']['vf_coef']),
            'max_grad_norm': round_hyperparam(params['params']['max_grad_norm']),
        }
    else:
        return {
            'n_steps': n_steps_list[int(params['params']['n_steps_index'])],
            'batch_size': batch_sizes[int(params['params']['batch_size_index'])],
            'n_epochs': int(params['params']['n_epochs']),
            'learning_rate': round_hyperparam(params['params']['learning_rate']),
            'clip_range': round_hyperparam(params['params']['clip_range']),
            'gae_lambda': round_hyperparam(params['params']['gae_lambda']),
            'ent_coef': round_hyperparam(params['params']['ent_coef']),
            'vf_coef': round_hyperparam(params['params']['vf_coef']),
            'max_grad_norm': round_hyperparam(params['params']['max_grad_norm']),
        }

# def create_financial_params(params):
#     def round_value(value):
#         if isinstance(value, (int, float)):
#             if value < 1:
#                 return round(value, 2)
#             else:
#                 return round(value)
#         return value

#     financial_params = {key: round_value(value) for key, value in params['params'].items()}
    
#     # Map interval index to actual interval value
#     interval_index = financial_params.get('interval', 0)
#     if isinstance(interval_index, (int, float)) and 0 <= interval_index < len(possible_intervals):
#         financial_params['interval'] = possible_intervals[int(interval_index)]
#     else:
#         raise ValueError('Interval was not found. Exiting training...')

#     financial_params['symbols'] = constant_params['symbols']
#     financial_params['min_collateral'] = constant_params['min_collateral']
#     return financial_params

# # Example usage
# selected_params = params2  # Choose the paramsX you want
# financial_params = create_financial_params(selected_params)

# # selected_params = params_1  # Choose the paramsX you want
# best_params = {
#     'target': 138,
#     'params': {
#         'cooldown_period': 31,
#         'initial_balance': 8456.457077245697,
#         'interval': '15m',
#         'leverage_max': 137.03358607280884,
#         'leverage_min': 9.891376672025666,
#         'limit': 969,
#         'risk_per_trade': 0.04382754866174893,
#         'total_timesteps': 99532,
#         'boost_leverage': True,
#         'normalize_reward': True,
#         'sl_atr_mult': 2.061126000761475,
#         'sl_percentage': 0.07169573918702689,
#         'tp_atr_mult': 5.476136261652576,
#         'tp_percentage': 0.31899459753057435,
#         'kelly_fraction': 0.6876666132463612,
#         'min_collateral': 50,
#         'n_steps': 1024,
#         'batch_size': 256,
#         'n_epochs': 7,
#         'learning_rate': 0.0009133615411706203,
#         'clip_range': 0.3192517863314879,
#         'gae_lambda': 0.8499729684264345,
#         'ent_coef': 0.02194219044742515,
#         'vf_coef': 0.7157108432325209,
#         'max_grad_norm': 0.8665530219819648
#     }
# }
selected_params = params_1



# combo scores
# ------------ # ------------ # ------------ # ------------ # ------------ # ------------
# param_1 = {'target': -11151.23, 'params': {'batch_size_index': 2.2344996366177448, 'clip_range': 0.4, 'cooldown_period': 1.2757568961350017, 'ent_coef': 0.1, 'gae_lambda': 0.8293511781634226, 'initial_balance': 5461.829767128424, 'interval': 0.9313010568883545, 'learning_rate': 0.01, 'leverage_max': 71.7165635736518, 'leverage_min': 11.511101696370114, 'limit': 709.7340489309194, 'max_grad_norm': 0.7796536502777316, 'n_epochs': 8.793794493057302, 'n_steps_index': 3.0, 'risk_per_trade': 0.2, 'total_timesteps': 36818.837237754895, 'vf_coef': 1.0}, 'output_dir': './logs/20240928-230657'}
# param_2 = {'target': -16408.185999999998, 'params': {'batch_size_index': 3.0, 'clip_range': 0.4, 'cooldown_period': 1.0487218771230231, 'ent_coef': 0.1, 'gae_lambda': 0.8144629896856063, 'initial_balance': 5472.674751162806, 'interval': 0.9140238057906884, 'learning_rate': 0.01, 'leverage_max': 76.80946023577121, 'leverage_min': 20.0, 'limit': 712.3878234809457, 'max_grad_norm': 0.7793033295129266, 'n_epochs': 7.225352404641897, 'n_steps_index': 3.0, 'risk_per_trade': 0.2, 'total_timesteps': 36805.8756374447, 'vf_coef': 1.0}, 'output_dir': './logs/20240928-231122'}
# param_3 = {'target': -17917.664, 'params': {'batch_size_index': 1.251066014107722, 'clip_range': 0.31609734803264744, 'cooldown_period': 1.0021731215295528, 'ent_coef': 0.030233257263183978, 'gae_lambda': 0.8293511781634226, 'initial_balance': 5461.692973843989, 'interval': 0.9313010568883545, 'learning_rate': 0.0034621516631600474, 'leverage_max': 71.5797716499871, 'leverage_min': 11.237517946063782, 'limit': 709.5972572016474, 'max_grad_norm': 0.7796536502777316, 'n_epochs': 8.52021074275097, 'n_steps_index': 2.6343523091728365, 'risk_per_trade': 0.01520364270760597, 'total_timesteps': 36818.700407136086, 'vf_coef': 0.4755743221304143}, 'output_dir': './logs/20240928-230114'}
# param_4 = {'target': -44266.526, 'params': {'batch_size_index': 3.0, 'clip_range': 0.4, 'cooldown_period': 12.456980792827556, 'ent_coef': 0.1, 'gae_lambda': 0.8289161674188265, 'initial_balance': 5467.420890609694, 'interval': 0.9310137708803484, 'learning_rate': 0.01, 'leverage_max': 77.30735636387745, 'leverage_min': 20.0, 'limit': 715.3249350284458, 'max_grad_norm': 0.7801824019308458, 'n_epochs': 19.975895764922598, 'n_steps_index': 3.0, 'risk_per_trade': 0.2, 'total_timesteps': 36824.42936644536, 'vf_coef': 1.0}, 'output_dir': './logs/20240928-230959'}
# param_5 = {'target': -147283.3, 'params': {'batch_size_index': 1.6760694853372549, 'clip_range': 0.14211608157857014, 'cooldown_period': 4.763928292612697, 'ent_coef': 0.08007445686755367, 'gae_lambda': 0.9936523151438795, 'initial_balance': 6567.120890796214, 'interval': 3.4616130783465704, 'learning_rate': 0.008765127631437422, 'leverage_max': 136.29886625550017, 'leverage_min': 2.61584001602578, 'limit': 519.5273916164412, 'max_grad_norm': 0.4188812936951982, 'n_epochs': 26.709847592594155, 'n_steps_index': 0.2950405014991503, 'risk_per_trade': 0.09001044875095991, 'total_timesteps': 48315.58120602008, 'vf_coef': 0.5798487564757154}, 'output_dir': './logs/20240928-230440'}