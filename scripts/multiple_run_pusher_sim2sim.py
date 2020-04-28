import argparse
from subprocess import Popen, PIPE, STDOUT
import glob
import os

# ===== User arguments =====
parser = argparse.ArgumentParser()
parser.add_argument('--vae', type=str, default=None)
parser.add_argument('--enable_render', action='store_true')

parser.add_argument('--result_path', type=str, default='results')
parser.add_argument('--n_test', type=int, default=5)
parser.add_argument('--exp', type=str, default='')
args = parser.parse_args()


def main():
    # ===== Predefined arguments =====
    file = 'scripts/run_pusher_sim2sim.py'
    policy_ckpt = 'data/04-14-dev-new-env-examples-skewfit-sawyer-push-SawyerPushNIPSEasy-v0/04-14-dev-new-env-examples-skewfit-sawyer-push-SawyerPushNIPSEasy-v0_2020_04_14_11_27_22_0000--s-10707/params.pkl'

    if not os.path.exists(args.result_path):
        print('[ERROR] The result_path is not exist:', args.result_path)

    if args.exp == '':
        print('[WARNING] You should set name of experiment.')
        args.exp = 'None'
    saved_dir = os.path.join(args.result_path, args.exp)
    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)

    vae_ckpt_list = glob.glob(args.vae + '*')
    cmds_list = [
        ['python', file, policy_ckpt, '--hide', '--n_test', str(args.n_test),
         '--result_path', os.path.join(args.result_path, args.exp),
         '--vae', vae_ckpt, '--exp', args.exp + '-' + vae_ckpt.split('-')[-1]] for vae_ckpt in vae_ckpt_list]

    # ===== Running parallel =====
    # procs = Popen(cmds_list[0], stdout=PIPE, stderr=PIPE)
    # procs.wait()
    # while True:
    #     output = procs.stdout.readline()
    #     if output == '' and procs.poll() is not None:
    #         break
    #     if output:
    #         print(output.strip())

    procs_list = [Popen(cmd, stdout=PIPE, stderr=PIPE) for cmd in cmds_list]
    n_procs, proc_cnt = len(procs_list), 0
    print('Running all experiments...')
    for proc in procs_list:
        proc.wait()
        # proc_cnt += 1
        # if proc_cnt == n_procs - 1:
        #     while True:
        #         output = proc.stdout.readline()
        #         if output == '' and proc.poll() is not None:
        #             break
        #         if output:
        #             print(output.strip())
    print('Complete running all experiments.')


if __name__ == '__main__':
    main()
