srcdir="$(dirname "{0%}")" # source directory
echo "$srcdir"
cd "$srcdir"/../arctic_flaming_monkey_typhoon || exit
git checkout v0.1.0 # change this to update dependency version referenced