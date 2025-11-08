ARG VITE_RAGZOOM_API_URL=http://api:8000

FROM node:20-slim AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM deps AS builder
ARG VITE_RAGZOOM_API_URL
ENV VITE_RAGZOOM_API_URL=${VITE_RAGZOOM_API_URL}
COPY . .
RUN npm run build

FROM deps AS dev
ARG VITE_RAGZOOM_API_URL
ENV VITE_RAGZOOM_API_URL=${VITE_RAGZOOM_API_URL}
COPY . .
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "4173"]

FROM nginx:1.25-alpine AS runner
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
