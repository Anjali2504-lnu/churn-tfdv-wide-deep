PY=python

.PHONY: install tfdv train test clean

install:
	$(PY) -m pip install -r requirements.txt

tfdv:
	mkdir -p artifacts/tfdv
	$(PY) src/data_validation.py --data_dir data --out_dir artifacts/tfdv

train:
	mkdir -p artifacts/model
	$(PY) src/train_model.py --data_dir data --out_dir artifacts/model --epochs 8 --batch_size 64

test:
	pytest -q

clean:
	rm -rf artifacts artifacts_test
