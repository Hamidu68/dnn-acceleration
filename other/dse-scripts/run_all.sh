function qsuball() {
        find -name "run.sh" -exec chmod +x {} \;
        find -name "run.sh" -exec qsub {} \;
}
qsuball
