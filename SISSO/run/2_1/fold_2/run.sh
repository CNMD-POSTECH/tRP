#!/bin/sh
#PBS -N -
#PBS -q full
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=28:host=n027:mpiprocs=28:ompthreads=1+1:ncpus=28:host=n028:mpiprocs=28:ompthreads=1+1:ncpus=28:host=n029:mpiprocs=28:ompthreads=1

module load intel/oneapi/2023.0.0
#EXE=vasp_std

NUMBER=`cat $PBS_NODEFILE | wc -l`

cd $PBS_O_WORKDIR

echo $PBS_JOBID `pwd` '    '  `date` >> ~/a.log/input.log

mpirun -machinefile $PBS_NODEFILE -np $NUMBER /home/gyrudgyrud21/0.File/sisso  >> relax.out

#Init_time=`sed -n '3p' OUTCAR | awk '{print $5, $6}'`
#Used_core=`sed -n '4p' OUTCAR | awk '{print $3}'`
#Cpu_time=`grep 'Total CPU' OUTCAR | awk '{print $6}'`

echo $PBS_JOBID >> ~/a.log/output.log

exit 0
