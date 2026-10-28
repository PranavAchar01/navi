// ESP32 belt emulator: runs firmware/belt/belt_core.h on the host. Same UDP protocol and port
// as the board, same 2 ms loop, same watchdog. Usage: belt-emu [port]   (default 4210)
#include <arpa/inet.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>
#include "../belt/belt_core.h"

static uint32_t millis(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint32_t)(ts.tv_sec * 1000ull + ts.tv_nsec / 1000000ull);
}

int main(int argc, char **argv) {
  int port = argc > 1 ? atoi(argv[1]) : 4210;
  int fd = socket(AF_INET, SOCK_DGRAM, 0);
  struct sockaddr_in me = {0}, peer = {0};
  me.sin_family = AF_INET;
  me.sin_port = htons(port);
  me.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (bind(fd, (struct sockaddr *)&me, sizeof me) < 0) {
    perror("bind");
    return 1;
  }
  fcntl(fd, F_SETFL, O_NONBLOCK);
  belt_t belt;
  belt_init(&belt);
  int havePeer = 0;
  uint32_t lastTel = 0;
  fprintf(stderr, "belt-emu: listening on 127.0.0.1:%d\n", port);
  for (;;) {
    uint32_t now = millis();
    uint8_t buf[64];
    struct sockaddr_in from;
    socklen_t fl = sizeof from;
    ssize_t n;
    while ((n = recvfrom(fd, buf, sizeof buf, 0, (struct sockaddr *)&from, &fl)) > 0) {
      if (belt_rx(&belt, buf, (int)n, now)) {
        peer = from;
        havePeer = 1;
      }
      fl = sizeof from;
    }
    uint8_t duty[BELT_N];
    belt_tick(&belt, now, duty);
    if (havePeer && now - lastTel >= 20) {
      uint8_t tel[BELT_TEL_LEN];
      belt_telemetry(&belt, now, duty, tel);
      sendto(fd, tel, sizeof tel, 0, (struct sockaddr *)&peer, sizeof peer);
      lastTel = now;
    }
    usleep(2000);
  }
}
