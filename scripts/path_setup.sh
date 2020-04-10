case "$(uname -s)" in
  Darwin)
    echo export $1=$2 >> ~/.bash_profile
    source ~/.bash_profile
    echo "updated: ~/.bash_profile"
  ;;
  Linux)
    echo export $1=$2 >> ~/.bashrc
    source ~/.bashrc
    echo "updated: ~/.bashrc"
  ;;
  *)
    echo 'Automatic path setup is only configured for MacOS and Linux.'
  ;;
esac
