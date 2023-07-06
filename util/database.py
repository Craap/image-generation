import sqlite3


class GelbooruDatabase:
    def __init__(self) -> None:
        self.con = sqlite3.connect("data/gelbooru.db")
        self.con.row_factory = sqlite3.Row
        self.cur = self.con.cursor()

        self.cur.execute("CREATE TABLE IF NOT EXISTS tag("
                         "  id INTEGER PRIMARY KEY,"
                         "  name TEXT,"
                         "  count INTEGER,"
                         "  type INTEGER,"
                         "  ambiguous INTEGER"
                         ")"
                         )

        self.cur.execute("CREATE TABLE IF NOT EXISTS post("
                         "  id INTEGER PRIMARY KEY,"
                         "  created_at TEXT,"
                         "  tags TEXT,"
                         "  score INTEGER,"
                         "  width INTEGER,"
                         "  height INTEGER,"
                         "  md5 TEXT,"
                         "  directory TEXT,"
                         "  image TEXT,"
                         "  rating TEXT,"
                         "  source TEXT,"
                         "  change INT,"
                         "  owner TEXT,"
                         "  creator_id INT,"
                         "  parent_id INT,"
                         "  sample INT,"
                         "  preview_width INT,"
                         "  preview_height INT,"
                         "  title TEXT,"
                         "  has_notes TEXT,"
                         "  has_comments TEXT,"
                         "  file_url TEXT,"
                         "  preview_url TEXT,"
                         "  sample_url TEXT,"
                         "  sample_height INT,"
                         "  sample_width INT,"
                         "  status TEXT,"
                         "  post_locked INT,"
                         "  has_children TEXT"
                         ")"
                         )
        
    def commit(self) -> None:
        self.con.commit()

    def insert_tags(self, tags: list) -> None:
        columns = tags[0].keys()
        query = f"INSERT OR IGNORE INTO tag({', '.join(columns)}) VALUES(:{', :'.join(columns)})"
        self.cur.executemany(query, tags)    

    def insert_posts(self, posts: list) -> None:
        columns = posts[0].keys()
        query = f"INSERT OR IGNORE INTO post({', '.join(columns)}) VALUES(:{', :'.join(columns)})"
        self.cur.executemany(query, posts)

    def get_posts(self) -> list[dict[str, any]]:
        return [dict(row) for row in self.cur.execute("SELECT * FROM post ORDER BY id").fetchall()]
    
    def get_tags(self) -> list[dict[str, any]]:
        return [dict(row) for row in self.cur.execute("SELECT * FROM tag ORDER BY id").fetchall()]