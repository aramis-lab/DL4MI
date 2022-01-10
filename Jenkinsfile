#!/usr/bin/env groovy

// Continuous Integration script to build Jupyter-Book
// Author: mauricio.diaz@inria.fr
pipeline {
  agent none
    stages {
      stage('Create Conda env') {
        agent { label 'gpu' }
        environment {
        PATH = "$HOME/miniconda/bin:$PATH"
        }
        //when { changeset "requirements.txt" }
        steps {
          echo 'Create Conda env for Jupyter-book'
          echo 'My branch name is ${BRANCH_NAME}'
          sh 'echo "My branch name is ${BRANCH_NAME}"'
          sh 'echo "Agent name: ${NODE_NAME}"'
          sh '''#!/usr/bin/env bash
             set +x
             eval "$(conda shell.bash hook)"
             conda create -y -n jb_env python=3.8 jupyterlab
             conda activate jb_env
             pip install -r jupyter-book/requirements.txt
             conda deactivate
             '''
        }
      }
      stage('Build') {
        agent { label 'gpu' }
        environment {
          PATH = "$HOME/miniconda/bin:$PATH"
          GIT_SSH_COMMAND = 'ssh -i /builds/.ssh/github_idrsa'
        }
        steps {
          echo 'Building Jupyter-book...'
          sh 'echo "Agent name: ${NODE_NAME}"'
          sh '''#!/usr/bin/env bash
             set +x
             eval "$(conda shell.bash hook)"
             conda activate jb_env
             make
             cd jupyter-book
             mkdir -p _build/.jupyter_cache
             make clean
             make
             sed -i 's+github/aramis-lab/DL4MI/blob/main/jupyter-book/notebooks+github/aramis-lab/DL4MI/blob/student22/notebooks+g' _build/html/notebooks/*.html
             conda deactivate
             '''
          stash(name: 'doc_html', includes: 'jupyter-book/_build/html/**')
        }
      }
      stage('Deploy') {
        agent { label 'gpu' }
        environment {
          PATH = "$HOME/miniconda/bin:$PATH"
        }
        steps {
          echo 'Deploying in webserver...'
          sh 'echo "Agent name: ${NODE_NAME}"'
          unstash(name: 'doc_html')
          sh '''#!/usr/bin/env bash
             set +x
             ls ./
             scp -r jupyter-book/_build/html/* aramislab.paris.inria.fr:~/workshops/DL4MI/2022/
             '''
          echo 'Finish uploading artifacts'
        }
      }
    }
//    post {
//      success {
//        mattermostSend(
//          color: "##A837C4",
//          message: "The tutorial has been updated, <https://aramislab.paris.inria.fr/workshops/DL4MI/intro.html|see here>"
//        )
//      }
//      failure {
//        mail to: 'clinicadl-ci@inria.fr',
//          subject: "Failed Pipeline: ${currentBuild.fullDisplayName}",
//          body: "Something is wrong with ${env.BUILD_URL}"
//      }
//    }
}
