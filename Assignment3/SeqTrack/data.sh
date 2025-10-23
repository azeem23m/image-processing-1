cd data/lasot
for z in *.zip; do
    unzip -q "$z" -d "${z%.zip}"   # extract into folder named after the zip file
done
