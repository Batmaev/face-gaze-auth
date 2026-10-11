FROM node:22-alpine AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

ARG VITE_API_URL=/api
ARG VITE_PROCESSING_TIME_SEC=25
ENV VITE_API_URL=$VITE_API_URL VITE_PROCESSING_TIME_SEC=$VITE_PROCESSING_TIME_SEC

RUN npm run build

FROM caddy:alpine

COPY Caddyfile /etc/caddy/Caddyfile
COPY --from=build /app/dist /srv
