tee ~/ML_course_assessment/local.yml << END
name: local
channels:
  - conda-forge
dependencies:
  - nbconvert


prefix: /home/jovyan/envs/ML_course_assessment

END

conda init bash
conda activate ML_course_assessment
conda env update --file local.yml --prune
