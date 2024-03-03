from app import app, db

# 创建数据库表格
with app.app_context():
    db.create_all()
