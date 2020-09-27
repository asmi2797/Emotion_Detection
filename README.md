# Emotion_Detection
Web Application of Emotion Detection done using HTML,CSS,Javascript,Python,FLask.
This project was deployed on Pycharm.

##Before running app.py
1) Make sure you install all the required packages stated in requirements.txt
2) Create folder named 'uploads' in static and enter the path on line no.35 in app.py
3) Config database connection accordingly.
4) Database name is flask and it contains 3 tables namely image,query and admin.

```
+----------+--------------+------+-----+-------------------+-------------------+
| Field    | Type         | Null | Key | Default           | Extra             |
+----------+--------------+------+-----+-------------------+-------------------+
| hash     | char(38)     | NO   | PRI | NULL              |                   |
| arrival  | datetime     | NO   |     | CURRENT_TIMESTAMP | DEFAULT_GENERATED |
| name     | varchar(30)  | YES  |     | NULL              |                   |
| detected | longblob     | NO   |     | NULL              |                   |
| faces    | int          | NO   |     | NULL              |                   |
| emotions | varchar(200) | NO   |     | NULL              |                   |
+----------+--------------+------+-----+-------------------+-------------------+```

-query
```+----------+-------------+------+-----+---------+----------------+
| Field    | Type        | Null | Key | Default | Extra          |
+----------+-------------+------+-----+---------+----------------+
| id       | int         | NO   | PRI | NULL    | auto_increment |
| name     | varchar(30) | NO   |     | NULL    |                |
| emailid  | varchar(20) | NO   |     | NULL    |                |
| phone    | varchar(10) | NO   |     | NULL    |                |
| comments | varchar(50) | NO   |     | NULL    |                |
+----------+-------------+------+-----+---------+----------------+```

-admin
```+----+----------+----------+
| id | username | password |
+----+----------+----------+
|  1 | admin    | admin123 |
+----+----------+----------+```
