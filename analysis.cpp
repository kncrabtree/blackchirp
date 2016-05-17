#include "analysis.h"

quint32 Analysis::nextPowerOf2(quint32 n)
{
    if(n == 0)
        return 1;

    n--;
    for(int i = 1; i<32; i*=2)
        n |= n >> i;

    return n+1;
}


qint64 Analysis::intRoundClosest(const qint64 n, const qint64 d)
{
    return ((n < 0) ^ (d < 0)) ? ((n - d/2)/d) : ((n + d/2)/d);
}


QList<int> Analysis::parseIntRanges(const QString str, int max)
{
    QList<int> out;

    QList<QString> l = str.split(QChar(','),QString::SkipEmptyParts);
    for(int i=0; i<l.size(); i++)
    {
        QString s = l.at(i).trimmed();
        if(s.contains(QChar('-')))
        {
            QList<QString> l2 = s.split(QChar('-'),QString::SkipEmptyParts);
            if(l2.size() != 2)
                continue;

            bool ok = false;
            int start = l2.at(0).trimmed().toInt(&ok);
            if(!ok || start < 1 || start > max)
                continue;
            int end = l2.at(1).trimmed().toInt(&ok);
            if(!ok || end < 1 || end > max)
                continue;

            if(end < start)
                std::swap(start,end);

            for(int j=start; j<=qMin(max,end); j++)
                out.append(j);
        }
        else
        {
            bool ok = false;
            int num = l.at(i).trimmed().toInt(&ok);
            if(!ok || num < 1 || num > max)
                continue;

            out.append(num);
        }
    }

    return out;
}
