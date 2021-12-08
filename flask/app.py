from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def basic():
    return render_template('basic.html')


@app.route('/result/', methods=['POST', 'GET'])
def result():
    city_list = ['강서구', '양천구', '구로구', '영등포구', '금천구', '동작구', '관악구', '서초구', '강남구', '송파구', '강동구', '은평구', '서대문구', '마포구',
                 '종로구', '중구', '용산구', '성동구', '광진구', '동대문구', '중랑구', '노원구', '도봉구', '강북구', '성북구']
    if request.method == 'POST':
        val = request.form['name']
        if val not in city_list:
            return render_template('error.html', index2=val)
        return render_template('result.html', index=val)

    return "hello"



if __name__ == '__main__':
    app.run(host="127.0.0.1", port="5000", debug=True)