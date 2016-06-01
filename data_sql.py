# -*- coding: utf-8 -*-

"""
data_sql
@author: kue
"""
import MySQLdb


class tianchi_data:
    def __init__(self):
        self.db_host = '127.0.0.1'
        self.db_user = 'root'
        self.db_passwd = '19920909'
        self.db_db = 'music_tianchi'
        self.db_port = 3306
        self.conn = None

    def connect(self):
        try:
            conn = MySQLdb.connect(host=self.db_host, user=self.db_user, passwd=self.db_passwd, db=self.db_db,
                                   port=self.db_port)
            self.conn = conn
        except MySQLdb.Error, e:
            print "Mysql Error %d: %s" % (e.args[0], e.args[1])

    def get_connection(self):
        return self.conn

    def close(self):
        if self.conn is not None:
            self.conn.close()

    def query(self, sql):
        self.connect()
        cursor = self.conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        self.close()
        return rows

    def insert_many(self,sql,args):
        self.connect()
        cursor = self.conn.cursor()
        try:
            cursor.executemany(sql, args)
        except Exception as e:
            print("执行MySQL: %s 时出错：%s" % (sql, e))
        finally:
            cursor.close()
            self.conn.commit()
            self.close()
