import os
import numpy as np
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str, help='path to the snapshot file')
parser.add_argument('--plot', action='store_true')
parser.add_argument('--gen', action='store_true')
parser.add_argument('--n_goals', type=int, default=30)
parser.add_argument('--id', type=int, default=None)
args_user = parser.parse_args()


def main():
    data_path = os.path.join(args_user.dir, 'results.npz')
    test_goals_file = os.path.join(args_user.dir, 'test_goal.txt')
    data = np.load(data_path)
    ee_coors_key = 'ee_coors'
    ob_coors_key = 'ob_coors'
    hand_key = 'hand_infos'
    puck_key = 'puck_infos'
    n_goals_per_cat = int(args_user.n_goals / 3)

    # ============== Rule to avoid goals ==============
    rule_obj_pos_1 = np.array([0.6, 0.0])       # Avoid object's goal at [0.6,   0.0]
    rule_obj_pos_2 = np.array([None, 0.0])      # Avoid object's goal at [*,     0.0]
    rule_obj_pos_3_1 = np.array([0.6, 0.05])    # Avoid object's goal at [0.6,   0.05]
    rule_obj_pos_3_2 = np.array([0.6, -0.05])   # Avoid object's goal at [0.6,  -0.05]
    rule_obj_pos_4_1 = np.array([0.65, 0.05])   # Avoid object's goal at [0.65,  0.05]
    rule_obj_pos_4_2 = np.array([0.65, -0.05])  # Avoid object's goal at [0.65, -0.05]
    rule_ee_dist_5 = 0.06                       # Avoid goals that ee's distance > 0.06
    rule_ee_vs_ob_dist_6 = 0.1                  # Avoid goals that ||ee's - obs's|| < 0.1
    rule_ob_dist_7 = 0.06                       # Avoid goals that obs's distance > 0.06
    rule_std_small_8 = 0.005                    # Avoid goals that std of obs's distance > 0.005

    n_data = data[ee_coors_key].shape[0]
    idxes_selected_ee_coors = []
    for i in range(n_data):
        rule_1 = not (data[ob_coors_key][i][0] == rule_obj_pos_1[0] and
                      data[ob_coors_key][i][1] == rule_obj_pos_1[1])
        rule_2 = data[ob_coors_key][i][1] != rule_obj_pos_2[1]
        rule_3 = (not (data[ob_coors_key][i][0] == rule_obj_pos_3_1[0] and
                       data[ob_coors_key][i][1] == rule_obj_pos_3_1[1]) and
                  not (data[ob_coors_key][i][0] == rule_obj_pos_3_2[0] and
                       data[ob_coors_key][i][1] == rule_obj_pos_3_2[1]))
        rule_4 = (not (data[ob_coors_key][i][0] == rule_obj_pos_4_1[0] and
                       data[ob_coors_key][i][1] == rule_obj_pos_4_1[1]) and
                  not (data[ob_coors_key][i][0] == rule_obj_pos_4_2[0] and
                       data[ob_coors_key][i][1] == rule_obj_pos_4_2[1]))
        rule_5 = data[hand_key][i][0] <= rule_ee_dist_5
        rule_6 = np.linalg.norm(data[ee_coors_key][i] - data[ob_coors_key][i]) >= rule_ee_vs_ob_dist_6
        rule_7 = data[puck_key][i][0] <= rule_ob_dist_7
        rule_8 = data[puck_key][i][1] <= rule_std_small_8
        rule_final = rule_1 and rule_2 and rule_3 and rule_4 \
                     and rule_5 and rule_6 and rule_7 and rule_8
        if rule_final:
            idxes_selected_ee_coors.append(i)

    # ============== Rule to select based on ranking ==============
    easy_goal_idxes = []
    medi_goal_idxes = []
    hard_goal_idxes = []
    for i in idxes_selected_ee_coors:
        easy_level = 0.02
        medi_level = 0.04
        hard_level = 0.06
        if data[puck_key][i][0] <= easy_level:
            easy_goal_idxes.append(i)
        elif easy_level < data[puck_key][i][0] <= medi_level:
            medi_goal_idxes.append(i)
        elif medi_level < data[puck_key][i][0] <= hard_level:
            hard_goal_idxes.append(i)

    # Sorting
    _easy_idxes = np.argsort(data[puck_key][easy_goal_idxes][:, 0])
    _medi_idxes = np.argsort(data[puck_key][medi_goal_idxes][:, 0])
    _hard_idxes = np.argsort(data[puck_key][hard_goal_idxes][:, 0])
    easy_goal_idxes = np.array(easy_goal_idxes)[_easy_idxes]
    medi_goal_idxes = np.array(medi_goal_idxes)[_medi_idxes]
    hard_goal_idxes = np.array(hard_goal_idxes)[_hard_idxes]
    n_easy = data[puck_key][easy_goal_idxes].shape[0]
    n_medi = data[puck_key][medi_goal_idxes].shape[0]
    n_hard = data[puck_key][hard_goal_idxes].shape[0]

    print('Number of easy goals = {}'.format(n_easy))
    print('Number of medi goals = {}'.format(n_medi))
    print('Number of hard goals = {}'.format(n_hard))
    print('Total of goals = {}\n'.format(n_easy + n_medi + n_hard))

    print('Easy idxes: {}'.format(easy_goal_idxes))
    print('Medi idxes: {}'.format(medi_goal_idxes))
    print('Hard idxes: {}\n'.format(hard_goal_idxes))

    if args_user.id is not None:
        print('ID={}, EE coor={}, OB coor={}, EE dist={}, OB dist={}'.format(args_user.id,
              data[ee_coors_key][args_user.id], data[ob_coors_key][args_user.id],
              data[hand_key][args_user.id], data[puck_key][args_user.id]))

    # ====================== Select randomly in selected goal pool ======================
    if args_user.gen:
        # final_easy_idxes = np.random.choice(easy_goal_idxes, n_goals_per_cat, replace=False)
        # final_medi_idxes = np.random.choice(medi_goal_idxes, n_goals_per_cat, replace=False)
        # final_hard_idxes = np.random.choice(hard_goal_idxes, n_goals_per_cat, replace=False)
        # Take first 10 sample of each category
        final_easy_idxes = easy_goal_idxes[:n_goals_per_cat] \
            if n_goals_per_cat <= len(easy_goal_idxes) else easy_goal_idxes
        final_medi_idxes = medi_goal_idxes[:n_goals_per_cat] \
            if n_goals_per_cat <= len(medi_goal_idxes) else medi_goal_idxes
        final_hard_idxes = hard_goal_idxes[:n_goals_per_cat] \
            if n_goals_per_cat <= len(hard_goal_idxes) else hard_goal_idxes

        # Always [EE, OBJ]
        set_of_goals = np.zeros((n_goals_per_cat * 3, 2, 2))
        set_of_goals[:n_goals_per_cat, 0] = data[ee_coors_key][final_easy_idxes]
        set_of_goals[:n_goals_per_cat, 1] = data[ob_coors_key][final_easy_idxes]
        set_of_goals[n_goals_per_cat:2*n_goals_per_cat, 0] = data[ee_coors_key][final_medi_idxes]
        set_of_goals[n_goals_per_cat:2*n_goals_per_cat, 1] = data[ob_coors_key][final_medi_idxes]
        set_of_goals[2*n_goals_per_cat:3*n_goals_per_cat, 0] = data[ee_coors_key][final_hard_idxes]
        set_of_goals[2*n_goals_per_cat:3*n_goals_per_cat, 1] = data[ob_coors_key][final_hard_idxes]

        f = open(test_goals_file, 'w')
        f.write('idxes_easy = {}\n'.format(final_easy_idxes))
        f.write('idxes_medi = {}\n'.format(final_medi_idxes))
        f.write('idxes_hard = {}\n'.format(final_hard_idxes))
        f.write('\nset_of_goals = np.array([\n')
        for i in range(3 * n_goals_per_cat):
            f.write('[[{}, {}], [{}, {}]],\n'.format(set_of_goals[i, 0, 0], set_of_goals[i, 0, 1],
                                                     set_of_goals[i, 1, 0], set_of_goals[i, 1, 1]))
        f.write('])')
        f.close()
    # ====================== Plotting ======================
    if args_user.plot:
        plt.figure()
        plt.plot(data[puck_key][easy_goal_idxes, 0])
        plt.fill_between(list(range(n_easy)),
                         data[puck_key][easy_goal_idxes, 0] + data[puck_key][easy_goal_idxes, 1],
                         data[puck_key][easy_goal_idxes, 0] - data[puck_key][easy_goal_idxes, 1],
                         alpha=0.25)
    
        plt.plot(data[puck_key][medi_goal_idxes, 0])
        plt.fill_between(list(range(n_medi)),
                         data[puck_key][medi_goal_idxes, 0] + data[puck_key][medi_goal_idxes, 1],
                         data[puck_key][medi_goal_idxes, 0] - data[puck_key][medi_goal_idxes, 1],
                         alpha=0.25)
        
        plt.plot(data[puck_key][hard_goal_idxes, 0])
        plt.fill_between(list(range(n_hard)),
                         data[puck_key][hard_goal_idxes, 0] + data[puck_key][hard_goal_idxes, 1],
                         data[puck_key][hard_goal_idxes, 0] - data[puck_key][hard_goal_idxes, 1],
                         alpha=0.25)
        plt.legend(['easy', 'medium', 'hard'])
        plt.ylabel('object\'s distance')
        plt.title('Object\'s distance')
    
        plt.figure()
        plt.plot(data[hand_key][easy_goal_idxes, 0])
        plt.fill_between(list(range(n_easy)),
                         data[hand_key][easy_goal_idxes, 0] + data[hand_key][easy_goal_idxes, 1],
                         data[hand_key][easy_goal_idxes, 0] - data[hand_key][easy_goal_idxes, 1],
                         alpha=0.25)
    
        plt.plot(data[hand_key][medi_goal_idxes, 0])
        plt.fill_between(list(range(n_medi)),
                         data[hand_key][medi_goal_idxes, 0] + data[hand_key][medi_goal_idxes, 1],
                         data[hand_key][medi_goal_idxes, 0] - data[hand_key][medi_goal_idxes, 1],
                         alpha=0.25)
    
        plt.plot(data[hand_key][hard_goal_idxes, 0])
        plt.fill_between(list(range(n_hard)),
                         data[hand_key][hard_goal_idxes, 0] + data[hand_key][hard_goal_idxes, 1],
                         data[hand_key][hard_goal_idxes, 0] - data[hand_key][hard_goal_idxes, 1],
                         alpha=0.25)
        plt.legend(['easy', 'medium', 'hard'])
        plt.ylabel('hand\'s distance')
        plt.title('Hand\'s distance')
    
        plt.show()


if __name__ == '__main__':
    main()
