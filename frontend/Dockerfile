# Dockerfile for React Frontend

# Aşama 1: Uygulamayı Derle (Build)
# Node.js'in resmi LTS (Uzun Süreli Destek) sürümünü kullan
FROM node:18-alpine AS builder

# Çalışma dizinini ayarla
WORKDIR /app

# package.json ve package-lock.json dosyalarını kopyala
# Bu adım, bağımlılıkların önbelleğe alınmasını sağlar
COPY package.json ./
COPY package-lock.json ./

# Node bağımlılıklarını yükle
# --force: Eğer package-lock.json'da bir çakışma olursa zorla yükle (genellikle gerekli değildir ama bazen faydalı olabilir)
RUN npm install --force

# Uygulama kaynak kodunu kopyala
COPY . .

# React uygulamasını üretim için derle
# Bu adım, uygulamanın statik HTML, CSS ve JavaScript dosyalarını oluşturur
RUN npm run build

# Aşama 2: Uygulamayı Bir Web Sunucusu ile Sun
# Hafif bir Nginx imajını temel olarak kullan
FROM nginx:alpine

# Nginx'in varsayılan yapılandırma dosyasını sil
RUN rm /etc/nginx/conf.d/default.conf

# Özel Nginx yapılandırma dosyamızı kopyala
# Bu dosya, Nginx'in React uygulamasının statik dosyalarını nasıl sunacağını belirler
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Derlenmiş React uygulamasının statik dosyalarını Nginx'in sunacağı dizine kopyala
COPY --from=builder /app/build /usr/share/nginx/html

# Nginx'i başlat
# Nginx varsayılan olarak 80 numaralı portta dinler
CMD ["nginx", "-g", "daemon off;"]
