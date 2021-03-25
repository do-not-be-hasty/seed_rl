from mrunner.helpers.specification_helper import create_experiments_helper
import os

tags = os.environ["PROJECT_TAG"].split(' ') if "PROJECT_TAG" in os.environ.keys() else []

experiments_list = create_experiments_helper(
    experiment_name='Test',
    base_config={
        'task_name': None,
        'learning_rate': 0.001,
    },
    #params_grid={},
    params_grid={'task_name': ['MMM2', '3s5z_vs_3s6z', '10m_vs_11m', 'corridor', '2c_vs_64zg'] * 4},
    #project=os.environ["NEPTUNE_PROJECT_NAME"],
    script='./seed_rl/docker/run_remote.sh starcraft vtrace_better 30',
    exclude=['.pytest_cache', '.git', 'docs', 'data', 'data_out', 'assets', 'out', '.vagrant', 'seed_rl/.git'],
    python_path=':ncc',
    with_mpi=False,
    tags=tags,
    callbacks=[],
    with_neptune=True,
    project_name='pmtest/marl-vtrace'
)
