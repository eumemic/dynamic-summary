# UI build stage
FROM node:20-slim AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
ARG VITE_RAGZOOM_API_URL=http://api:8000
ENV VITE_RAGZOOM_API_URL=$VITE_RAGZOOM_API_URL
RUN npm run build

# Runtime image using nginx to serve static files
FROM nginx:1.25-alpine AS runner
COPY --from=builder /app/dist /usr/share/nginx/html
# Expose UI port
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
