#ifndef SEQUENCEMANAGER_H
#define SEQUENCEMANAGER_H

#include <QObject>

class SequenceManager : public QObject
{
    Q_OBJECT
public:
    explicit SequenceManager(QObject *parent = 0);
    ~SequenceManager();

signals:

public slots:
};

#endif // SEQUENCEMANAGER_H
