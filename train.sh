python -m train --trainset "data/pMSA_PF00207_PF07677_train.csv" \
			    --valset "data/pMSA_PF00207_PF07677_val.csv" \
			    --save "models/saved_PF00207_PF07677.pth.tar" \
				--load "" \
			    --modelconfig "shallow.config.json" \
				--outputfile "output.txt"
