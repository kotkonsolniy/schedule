from decouple import config
import psycopg2

CREATE_TEACHERS_TABLE = """
CREATE TABLE IF NOT EXISTS teachers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL
);
"""

# a = """
#     id_subjects PRIMARY
#     CONSTRAINT teacher_subject
#         FOREIGN KEY (id)
#             REFERENCES subjects(id)
#     CONSTRAINT teacher_time
#         FOREIGN KEY (id)
#             REFERENCES times(id);
# """

CREATE_GROUPS_TABLE = """
CREATE TABLE IF NOT EXISTS groups (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL
);
"""
# b = """
#     CONSTRAINT group_subject
#         FOREIGN KEY (id)
#             REFERENCES subjects(id);
# """


CREATE_SUBJECTS_TABLE = """
CREATE TABLE IF NOT EXISTS subjects (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL
);
"""

CREATE_TIMES_TABLE = """
CREATE TABLE IF NOT EXISTS times (
    id SERIAL PRIMARY KEY,
    name TIMESTAMP NOT NULL
);
"""

CREATE_CLASSROOMS_TABLE = """
CREATE TABLE IF NOT EXISTS classroom (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL
);
"""

# CREATE_DEPEND = """
# ALTER TABLE teachers
# ADD CONSTRAINT teacher_subject
# FOREIGN KEY (id)
# REFERENCES subjects(id)
# ADD CONSTRAINT teacher_time
# FOREIGN KEY (id)
# REFERENCES times(id)
#
# ALTER TABLE groups
# ADD CONSTRAINT group_subject
# FOREIGN KEY (id)
# REFERENCES subjects(id)
#
# ALTER TABLE subjects
# ADD CONSTRAINT subject_classroom
# FOREIGN KEY(id)
# REFERENCES classrooms(id);
#"""
class Connect:
    POSTGRES_DB = config('POSTGRES_DB')
    POSTGRES_USER = config('POSTGRES_USER')
    POSTGRES_PASSWORD = config('POSTGRES_PASSWORD')
    POSTGRES_HOST = config('POSTGRES_HOST')
    POSTGRES_PORT = config('POSTGRES_PORT')

    def start_postgres(self):
        try:
            conn = psycopg2.connect(
                database=self.POSTGRES_DB,
                user=self.POSTGRES_USER,
                password=self.POSTGRES_PASSWORD,
                host=self.POSTGRES_HOST,
                port=self.POSTGRES_PORT
            )
            print('Подключение успешно')
            cur = conn.cursor()
            try:
                cur.execute(CREATE_TEACHERS_TABLE)
                cur.execute(CREATE_GROUPS_TABLE)
                cur.execute(CREATE_SUBJECTS_TABLE)
                cur.execute(CREATE_TIMES_TABLE)
                cur.execute(CREATE_CLASSROOMS_TABLE)
                print("Таблицы успешно созданы или уже существуют.\nЗависимости установлены")
                conn.commit()
                cur.close()
                conn.close()
                return conn
            except Exception as e:
                print(f'Ошибка при создании таблиц {e}')
                return False
        except psycopg2 as e:
            print(f'Ошибка подключения: {e}')
            return False


def main():
    conn = Connect()
    conn.start_postgres()
    # print("Connect")

if __name__ == '__main__':
    main()
