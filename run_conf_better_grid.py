from mrunner.helpers.specification_helper import create_experiments_helper
import os

tags = os.environ["PROJECT_TAG"].split(' ') if "PROJECT_TAG" in os.environ.keys() else []

experiments_list = create_experiments_helper(
    experiment_name='Test',
    base_config={
        'task_name': None,
        # Parameters of dataset:
    },
    params_grid={'task_name': ['2s_vs_1sc', '2s3z', '3s5z', '1c3s5z', '10m_vs_11m',
                          '2c_vs_64zg', 'bane_vs_bane', '5m_vs_6m', '3s_vs_5z',
                          '3s5z_vs_3s6z', '6h_vs_8z', '27m_vs_30m', 'MMM2', 'corridor']*3},
    #project=os.environ["NEPTUNE_PROJECT_NAME"],
    script='./seed_rl/docker/run_remote.sh starcraft vtrace_better 30',
    exclude=['.pytest_cache', '.git', 'docs', 'data', 'data_out', 'assets', 'out', '.vagrant'],
    python_path=':ncc',
    with_mpi=False,
    tags=tags,
    callbacks=[],
    with_neptune=True,
    project_name='pmtest/marl-vtrace'
)
