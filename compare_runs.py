from itertools import combinations
import datetime
import numpy as np
import pandas as pd

from evaluation_utils import one_fight

normal_easy = '/cs/ep/106/AI/run_trains/fc_s200_dis095_pun-10_fix_140721_194217/final_model'
normal_hard = '/cs/ep/106/AI/run_trains_v3/fc_one_300_ooo_f_180721_103644/s3f_ooo/model_5099'
old = 'old'
curriculum_size = '/cs/ep/106/AI/run_trains/fc_curric_size_dis0999_pun-10_fix_140721_181724/s400_model'
curriculum_player = '/cs/ep/106/AI/run_trains_v3/fc_2f_3f_3fr_3f22_3rr_3ro_3roo_180721_131305/final_model'
conv_player = '/cs/ep/106/AI/run_trains_v3/conv_s300_200721_170300/s300/model_4199'

all_players = [old, normal_easy, normal_hard, curriculum_size, curriculum_player]
all_players_names = ['old', 'normal_easy', 'normal_hard', 'curriculum_size', 'curriculum_player']

res = np.zeros((len(all_players), len(all_players)))

for p1, p2 in combinations(range(len(all_players)), 2):
    curr_res = one_fight([all_players[p1], all_players[p2]], num_of_fights=100)
    res[p1, p2] = curr_res[0]
    res[p2, p1] = curr_res[1]

df = pd.DataFrame(res, columns=all_players_names, index=all_players_names)
time = datetime.datetime.now().strftime("%d%m%y_%H%M%S")
df.to_csv(f'compare_runs_{time}.csv')


