`
python3 -m venv venv
source venv/bin/activate 
`

to start
`
pip install -r requirements.txt                      
uvicorn app.main:app --reload
`
