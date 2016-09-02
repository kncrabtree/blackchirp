#include "pico2206b.h"

Pico2206B::Pico2206B(QObject *parent) : MotorOscilloscope(parent)
{
    d_subKey = QString("pico2206b");
    d_prettyName = QString("Pico 2206B Oscilloscope");
    d_threaded = false;
    d_commType = CommunicationProtocol::Custom;
}



bool Pico2206B::testConnection()
{
}

void Pico2206B::initialize()
{
}

bool Pico2206B::configure(const BlackChirp::MotorScopeConfig &sc)
{
}

MotorScan Pico2206B::prepareForMotorScan(MotorScan s)
{
}
