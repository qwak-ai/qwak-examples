build:
	qwak models build . --model-id credit_risk_streaming  --main-dir main

deploy:
	qwak models deploy realtime --model-id credicredit_risk_streaming --pods 1

features:
	qwak features register -p main/features.py