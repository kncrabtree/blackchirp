#ifndef CLOCKMANAGER_H
#define CLOCKMANAGER_H

#include <QObject>


//here we need to select which hardware files to include for each clock


/**
 * @brief The ClockManager class associates hardware clocks with their purposes
 *
 * CP-FTMW experiments use a variety of clock frequencies. In simple cases, there
 * may be a PLDRO that is used to mix the chirp up and/or down in frequency, but in
 * more complex cases mutlitple clocks can be used to perform upconversion, downconversion,
 * AWG sample clocking, digitizer clocking, or even more.
 *
 * The ClockManager class is intended to keep track of which hardware clock serves each
 * purpose. Some clock objects (e.g., Valon 5009) have multiple independent outputs, so the
 * ClockManager may associate multiple functions with a single hardware object.
 *
 */

class ClockManager : public QObject
{
    Q_OBJECT
public:
    explicit ClockManager(QObject *parent = nullptr);

signals:

public slots:

private:

};

#endif // CLOCKMANAGER_H
