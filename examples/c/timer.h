#include <stdint.h>
#include <stdio.h>

inline uint64_t get_clock() {
    uint32_t lo, hi;
    __asm__ __volatile__ (
      "xorl %%eax, %%eax\n"
      "cpuid\n"
      "rdtsc\n"
      : "=a" (lo), "=d" (hi)
      :
      : "%ebx", "%ecx");
    return (uint64_t)hi << 32 | lo;
}

inline float get_frequency() {
  float cpu_speed = 0.0;
  char line[40];
  FILE *fh = fopen("/proc/cpuinfo", "rt");
  while (fgets(line, 40, fh)) {
      sscanf(line, "cpu MHz       : %f", &cpu_speed);
  }
  return cpu_speed;
}
