#define _CRT_SECURE_NO_WARNINGS 1
#define _WINSOCK_DEPRECATED_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <time.h>

#pragma comment(lib, "ws2_32.lib")

// TFTP 操作码，按照任务书上写的
#define SERVER_PORT 69
#define BUFFER_SIZE 516//加了首部
#define LOG_FILE "tftp_server1.log"//记录日志
#define MAX_RETRIES 5//最大重传次数
#define RRQ 1
#define WRQ 2
#define DATA 3
#define ACK 4
#define ERROR_CODE 5

typedef struct {
    char mode[10];
    char filename[256];
} RequestPacket;

typedef struct {
    short opcode;
    short blockNumber;
    char data[BUFFER_SIZE - 4];//操作码+块编号=4
} DataPacket;

typedef struct {
    short opcode;
    short blockNumber;
} AckPacket;

void error(const char* msg) {
    perror(msg);
    WSACleanup();
    exit(1);
}

// 日志记录，即日志函数处理部分
void log_message(const char* message) {
    FILE* logFile = fopen(LOG_FILE, "a");
    if (logFile) {
        time_t now;
        time(&now);
        fprintf(logFile, "[%s] %s\n", ctime(&now), message);
        fclose(logFile);
    }
}

// 发送错误信息，即错误信息处理部分
void send_error(SOCKET sock, struct sockaddr_in* client, int errorCode, const char* errorMsg) {
    char errorBuffer[BUFFER_SIZE];
    int clientLen = sizeof(*client);
    *(short*)errorBuffer = htons(ERROR_CODE);
    *(short*)(errorBuffer + 2) = htons(errorCode);
    strcpy(errorBuffer + 4, errorMsg);
    sendto(sock, errorBuffer, 4 + strlen(errorMsg) + 1, 0, (struct sockaddr*)client, clientLen);
    log_message(errorMsg);
    printf("%s\n", errorMsg);
}

void send_ack(SOCKET sock, struct sockaddr_in* client, short blockNumber) {
    AckPacket ack = { htons(ACK), htons(blockNumber) };
    sendto(sock, &ack, sizeof(AckPacket), 0, (struct sockaddr*)client, sizeof(*client));
}

// 上传处理 WRQ，即上传处理函数部分
void handleUpload(SOCKET sockfd, char* filename, struct sockaddr_in clientAddr, int isNetascii) {
    FILE* file = isNetascii ? fopen(filename, "w") : fopen(filename, "wb");
    if (!file) {
        send_error(sockfd, &clientAddr, 2, "Failed to open file for writing");
        return;
    }

    int retries = 0;
    int clientAddrLen = sizeof(clientAddr);
    short blockNumber = 0;
    char buffer[BUFFER_SIZE];
    int totalBytesTransferred = 0;

    send_ack(sockfd, &clientAddr, blockNumber);
    blockNumber++;

    while (1) {//接收数据包
        int n = recvfrom(sockfd, buffer, BUFFER_SIZE, 0, (struct sockaddr*)&clientAddr, &clientAddrLen);
        if (n < 0 && ++retries > MAX_RETRIES) {//超时或丢包
            log_message("Upload failed: Max retries exceeded");
            fclose(file);
            return;
        }

        DataPacket* dataPacket = (DataPacket*)buffer;
        int dataSize = n - 4;

        if (isNetascii) {//同样判断模式
            for (int i = 0; i < dataSize; i++) {
                if (dataPacket->data[i] == '\n') {
                    fputc('\r\n', file);
                    i++;
                }
                else if (dataPacket->data[i] == '\r') {
                    fputc('\r\0', file);
                    i++;
                }
                else {
                    fputc(dataPacket->data[i], file);
                }
            }
        }
        else {
            fwrite(dataPacket->data, 1, dataSize, file);
        }

        totalBytesTransferred += dataSize;

        send_ack(sockfd, &clientAddr, blockNumber);
        blockNumber++;

        if (dataSize < 512) break;
    }

    fclose(file);
    char logBuffer[100];
    snprintf(logBuffer, sizeof(logBuffer), "Upload completed: %d bytes", totalBytesTransferred);
    log_message(logBuffer);
    printf("Upload successful: %d bytes\n", totalBytesTransferred);
}

// 下载处理 RRQ，即下载函数部分
void handleDownload(SOCKET sockfd, char* filename, struct sockaddr_in clientAddr, int isNetascii) {
    FILE* file = isNetascii ? fopen(filename, "r") : fopen(filename, "rb");
    if (!file) {
        send_error(sockfd, &clientAddr, 1, "Read-requested file not found");
        return;
    }

    int retries = 0;
    int clientAddrLen = sizeof(clientAddr);
    DataPacket dataPacket;
    short blockNumber = 1;
    int totalBytesTransferred = 0;

    while (!feof(file)) {//发送数据包
        int bytesRead = 0;
        if (isNetascii) {
            char ch;//总接收字节数减去头部 
            while (bytesRead < BUFFER_SIZE - 4 && (ch = fgetc(file)) != EOF) {
                if (ch == '\n') {
                    dataPacket.data[bytesRead++] = '\r';
                    dataPacket.data[bytesRead++] = '\n';
                }
                else if (ch == '\r') {
                    dataPacket.data[bytesRead++] = '\r';
                    dataPacket.data[bytesRead++] = '\0';
                }
                else {
                    dataPacket.data[bytesRead++] = ch;
                }//判断模式，ascII就对数据进行换行符等特殊处理后写入文件；如果不是，则直接用fwrite函数将数据写入文件
            }
        }
        else {
            bytesRead = fread(dataPacket.data, 1, BUFFER_SIZE - 4, file);
        }

        dataPacket.opcode = htons(DATA);
        dataPacket.blockNumber = htons(blockNumber);

        sendto(sockfd, &dataPacket, bytesRead + 4, 0, (struct sockaddr*)&clientAddr, sizeof(clientAddr));

        AckPacket ack;
        while (1) {
            int n = recvfrom(sockfd, &ack, sizeof(ack), 0, (struct sockaddr*)&clientAddr, &clientAddrLen);
            if (n >= 0 && ntohs(ack.opcode) == ACK && ntohs(ack.blockNumber) == blockNumber) {
                break;
            }

            if (++retries > MAX_RETRIES) {
                log_message("Download failed: Max retries exceeded");
                fclose(file);
                return;
            }

            log_message("ACK lost: Retrying...");
            sendto(sockfd, &dataPacket, bytesRead + 4, 0, (struct sockaddr*)&clientAddr, clientAddrLen);
        }

        blockNumber++;
        totalBytesTransferred += bytesRead;
        if (bytesRead < 512) break;//数据大小小于 512 字节，最后一个数据包
    }

    fclose(file);
    char logBuffer[100];
    snprintf(logBuffer, sizeof(logBuffer), "Download completed: %d bytes", totalBytesTransferred);
    log_message(logBuffer);
    printf("Download successful: %d bytes\n", totalBytesTransferred);
}

// 处理请求并根据操作码决定上传或下载
void handle_request(SOCKET sock, struct sockaddr_in* client, char* buffer) {
    int opcode = ntohs(*(short*)buffer);
    char filename[256], mode[10];
    char* ptr = buffer + 2;

    strncpy(filename, ptr, sizeof(filename) - 1);
    ptr += strlen(filename) + 1;
    strncpy(mode, ptr, sizeof(mode) - 1);
    int isNetascii = (strcmp(mode, "netascii") == 0);

    //判断操作码
    if (opcode == RRQ) {
        log_message("Receiving a Download_request from client");
        handleDownload(sock, filename, *client, isNetascii);
    }
    else if (opcode == WRQ) {
        log_message("Receiving an Upload_request from client");
        handleUpload(sock, filename, *client, isNetascii);
    }
    else {
        send_error(sock, client, 4, "Invalid request");
    }
}

// 主程序启动 TFTP 服务器
int main() {
    WSADATA wsaData;
    SOCKET sockfd;
    struct sockaddr_in serverAddr, clientAddr;
    char buffer[BUFFER_SIZE];
    int clientAddrLen = sizeof(clientAddr);

    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        perror("WSAStartup failed");
        return 1;
    }

    sockfd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);//创建套接字
    if (sockfd == INVALID_SOCKET) {
        perror("Socket creation failed");
        WSACleanup();
        return 1;
    }

    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons(SERVER_PORT);
    //绑定服务器
    if (bind(sockfd, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
        perror("Binding failed");
        closesocket(sockfd);
        WSACleanup();
        return 1;
    }

    log_message("TFTP server started.");
    printf("TFTP server started.\n");

    while (1) {//循环接收指令
        int n = recvfrom(sockfd, buffer, BUFFER_SIZE, 0, (struct sockaddr*)&clientAddr, &clientAddrLen);
        if (n > 0) {
            handle_request(sockfd, &clientAddr, buffer);
        }
    }

    closesocket(sockfd);
    WSACleanup();
    return 0;
}