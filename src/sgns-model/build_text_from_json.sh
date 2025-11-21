jq < test.json ".text" | sed -e 's/^"//' -e 's/"$//' > test.txt
