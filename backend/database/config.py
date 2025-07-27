from databases import Database
from sqlalchemy import create_engine, MetaData
from .models import Base 
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:password@localhost/yolo_counter_db")

database = Database(DATABASE_URL)

# SQLAlchemy engine'i (migration'lar ve table oluşturma için)
# databases kütüphanesi doğrudan table oluşturmayı desteklemediği için sqlalchemy engine kullanıyoruz.
engine = create_engine(str(DATABASE_URL.replace("+asyncpg", "")), echo=False)

async def create_db_tables():
    print("Veritabanı tabloları oluşturuluyor...")
    # SQLAlchemy'nin MetaData'sını kullanarak tabloları oluşturun
    # Bu, henüz alembic gibi bir migration aracı kullanmadığımız için basit bir yoldur.
    # Üretim ortamında alembic gibi bir migration aracı kullanmak daha iyidir.
    Base.metadata.create_all(engine)
    print("Veritabanı tabloları oluşturuldu (veya zaten mevcut).")


async def connect_db():
    print("Veritabanına bağlanılıyor...")
    await database.connect()
    print("Veritabanına başarıyla bağlanıldı.")

async def disconnect_db():
    print("Veritabanından bağlantı kesiliyor...")
    await database.disconnect()
    print("Veritabanından bağlantı kesildi.")