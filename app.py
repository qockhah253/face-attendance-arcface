from flask import Flask, render_template, request, jsonify
import cv2, numpy as np, base64
from face_model import FaceRecognizer
from database import (init_db, save_student, get_all_students,
                      save_attendance, get_attendance_history,
                      get_students_list, delete_student)

app        = Flask(__name__)
recognizer = FaceRecognizer('models/arcface_epoch20.pth')
init_db()

def decode_image(b64):
    if ',' in b64:
        b64 = b64.split(',')[1]
    arr = np.frombuffer(base64.b64decode(b64), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

@app.route('/')
def index():
    students = get_students_list()
    return render_template('index.html', total=len(students))

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/attendance')
def attendance():
    return render_template('attendance.html')

@app.route('/history')
def history():
    return render_template('history.html')

@app.route('/students')
def students():
    return render_template('students.html', students=get_students_list())

@app.route('/api/register', methods=['POST'])
def api_register():
    data       = request.json
    name       = data.get('name', '').strip()
    student_id = data.get('student_id', '').strip()
    class_name = data.get('class_name', '').strip()
    image_b64  = data.get('image')
    if not all([name, student_id, image_b64]):
        return jsonify({'success': False, 'message': 'Vui lòng điền đầy đủ thông tin'})
    img  = decode_image(image_b64)
    face, _ = recognizer.detect_face(img)
    if face is None:
        return jsonify({'success': False, 'message': 'Không phát hiện khuôn mặt — vui lòng thử lại'})
    emb = recognizer.get_embedding(face)
    ok, msg = save_student(name, student_id, class_name, emb)
    return jsonify({'success': ok, 'message': msg})

@app.route('/api/attendance', methods=['POST'])
def api_attendance():
    data      = request.json
    image_b64 = data.get('image')
    is_late   = data.get('late', False)
    if not image_b64:
        return jsonify({'success': False, 'message': 'Không có ảnh'})
    img     = decode_image(image_b64)
    gallery = get_all_students()
    student, score, _, error = recognizer.recognize(img, gallery)
    if student is None:
        return jsonify({'success': False, 'message': error})
    status  = 'muộn' if is_late else 'đúng giờ'
    ok, msg = save_attendance(student['student_id'], student['name'],
                              student.get('class_name', ''), status)
    return jsonify({
        'success':    True,
        'already':    not ok,
        'message':    msg,
        'name':       student['name'],
        'student_id': student['student_id'],
        'class_name': student.get('class_name', ''),
        'score':      round(score * 100, 1),
        'status':     status
    })

@app.route('/api/history')
def api_history():
    date       = request.args.get('date')
    student_id = request.args.get('student_id')
    return jsonify(get_attendance_history(date, student_id))

@app.route('/api/students')
def api_students():
    return jsonify(get_students_list())

@app.route('/api/students/<student_id>', methods=['DELETE'])
def api_delete_student(student_id):
    delete_student(student_id)
    return jsonify({'success': True})

if __name__ == '__main__':
    print(" Server chạy tại: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
