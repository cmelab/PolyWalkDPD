"""Define the project's workflow logic and operation functions.

Execute this script directly from the command line, to view your project's
status, execute operations and submit them to a cluster. See also:

    $ python src/project.py --help
"""
import signac
import pickle
from flow import FlowProject, directives
from flow.environment import DefaultSlurmEnvironment
import os
from unyt import Unit


class DPD(FlowProject):
    pass


class Borah(DefaultSlurmEnvironment):
    hostname_pattern = "borah"
    template = "borah.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition",
            default="shortgpu-v100",
            help="Specify the partition to submit to."
        )


class Fry(DefaultSlurmEnvironment):
    hostname_pattern = "fry"
    template = "fry.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition",
            default= "batch",
            help="Specify the partition to submit to."
        )

@DPD.operation(
    directives={"ngpu": 1, "ncpu": 16, "executable": "python -u"}, name="run"
)
def run(job):
    """Run initial single-chain simulation."""
    import create_system_dpd
    from create_system_dpd import create_polymer_system_dpd as run
    import numpy as np
    with job:
        print("------------------------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("------------------------------------")

        pos, time = run(num_pol=job.sp.num_pol,num_mon=job.sp.num_mon,density=job.sp.density,A=job.sp.A,gamma=job.sp.gamma,k=job.sp.k,r_cut=job.sp.r_cut,seed=job.sp.seed)
        job.doc.time = time
       
if __name__ == "__main__":
    DPD(environment=Fry).main()
