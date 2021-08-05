tee ~/ML_course_assessment/ML_course_assessment.yml << END
name: ML_course_assessment
channels:
  - conda-forge
dependencies:
  - python==3.9
  - numpy
  - pandas
  - scipy
  - statsmodels
  - pingouin
  - scikit-learn
  - matplotlib
  - seaborn
  - plotly
  - pip
  - ipykernel
  - nb_conda_kernels
  - pip: 
    - nibabel

prefix: /home/jovyan/envs/ML_course_assessment

END

conda env create --file ~/ML_course_assessment/ML_course_assessment.yml
conda activate ML_course_assessment
python -m ipykernel install --user --name ML_course_assessment --display-name "Python (ML_course_assessment)"
