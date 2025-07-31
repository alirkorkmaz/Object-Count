from fastapi import WebSocket, WebSocketDisconnect


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"Yeni WebSocket bağlantısı: {websocket.client.host}:{websocket.client.port}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"WebSocket bağlantısı kesildi: {websocket.client.host}:{websocket.client.port}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except WebSocketDisconnect:
                disconnected.append(connection)
                print("WebSocket bağlantısı koptu, temizleniyor.")
            except Exception as e:
                print(f"WebSocket mesaj gönderim hatası: {e}")
                disconnected.append(connection)
        for ws in disconnected:
            if ws in self.active_connections:
                self.active_connections.remove(ws)

# Uygulama genelinde kullanılacak tekil yöneticiyi tanımlıyoruz
manager = ConnectionManager()
