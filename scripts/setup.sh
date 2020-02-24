cd .. || exit
cd ./arctic_flaming_monkey_typhoon || exit
git submodule update --init --recursive
git checkout v0.2.0 # change this to update dependency version referenced
