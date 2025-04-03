from pathlib import Path
from typing import List, Dict
import sqlite3


def init():
    print('start...')
    description = """ This script will read in all the patients from the inclusion list
        and make a table. Then each patient will be added and we'll use this database for storing
        the UQ data. """
    print(description)


def get_ids(inclusion_fpath: Path, debug: bool = False) -> List[Dict[str, str]]:
    """
    Read patient IDs from the inclusion file and return a list of dictionaries with patient data.

    Args:
        inclusion_fpath (Path): Path to the inclusion file.
        debug (bool): If True, print the first three patient data entries.

    Returns:
        List[Dict[str, str]]: List of dictionaries containing patient data.
    """
    with inclusion_fpath.open('r') as f:
        patients = f.readlines()

    pat_data = [{'id': pat_id.strip(), 'seq_id': pat_id.strip().split('_')[0], 'anon_id': pat_id.strip().split('_')[1]} for pat_id in patients]

    if debug:
        for pat in pat_data[:3]:
            print(pat)
    
    return pat_data


def make_table(conn: sqlite3.Connection, tablename: str = 'patients_uq'):
    """
    Create a table in the database to store patient data.

    Args:
        conn (sqlite3.Connection): Connection object to the database.
        tablename (str): Name of the table to create.
    """
    try:
        c = conn.cursor()
        c.execute(f'''CREATE TABLE IF NOT EXISTS {tablename}
                     (id text, seq_id text, anon_id text)''')
        conn.commit()
        print(f'Table "{tablename}" created.')
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    except Exception as e:
        print(f"Error creating table: {e}")


def insert_in_table(conn: sqlite3.Connection, ids: List[Dict[str, str]], tablename: str = 'patients_uq'):
    """
    Insert patient data into the table.

    Args:
        conn (sqlite3.Connection): Connection object to the database.
        ids (List[Dict[str, str]]): List of dictionaries containing patient data.
        tablename (str): Name of the table to insert data into.
    """
    try:
        c = conn.cursor()
        for pat in ids:
            c.execute(f"INSERT INTO {tablename} (id, seq_id, anon_id) VALUES (?, ?, ?)", (pat['id'], pat['seq_id'], pat['anon_id']))
        conn.commit()
        print(f'{len(ids)} patients inserted into table "{tablename}".')
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    except Exception as e:
        print(f"Error inserting data into table: {e}")


def connect_to_db(fname: Path) -> sqlite3.Connection:
    """
    Connect to the database.

    Args:
        fname (Path): Path to the database file.

    Returns:
        sqlite3.Connection: Connection object.
    """
    try:
        conn = sqlite3.connect(str(fname))
        return conn
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    except Exception as e:
        print(f"Error connecting to database: {e}")


def delete_table(conn: sqlite3.Connection, tablename: str = None):
    """
    Delete the table from the database.

    Args:
        conn (sqlite3.Connection): Connection object to the database.
        tablename (str): Name of the table to delete.
    """
    assert tablename is not None, 'Table name must be provided.'

    try:
        c = conn.cursor()
        c.execute(f"DROP TABLE IF EXISTS {tablename}")
        conn.commit()
        print(f'Table "{tablename}" deleted.')
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    except Exception as e:
        print(f"Error deleting table: {e}")


if __name__ == '__main__':

    inclusion_ids_fpath = Path('lists/inclusion/include_ids_sorted.lst')
    db_path             = Path('databases/master_habrok_20231106_v2.db')
    tablename           = 'patients_uq'

    # Step 1 - Read in the inclusion list
    ids = get_ids(inclusion_ids_fpath, debug=True)

    # step 2 - Connect to the database
    conn = connect_to_db(fname=db_path)

    # Step 2.1 - Delete the whole table if it exists
    delete_table(conn, tablename=tablename)

    # Step 3 - Make a table for the database - only function call that we will make
    make_table(conn, tablename=tablename)

    # step 4 - Insert the patients into the table
    insert_in_table(conn, ids, tablename=tablename)

    # Close the connection
    conn.close()

    print('done.')
