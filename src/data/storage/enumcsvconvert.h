#ifndef ENUMCSVCONVERT_H
#define ENUMCSVCONVERT_H

#include <QVariant>
#include <QMetaEnum>

/*!
 * \file enumcsvconvert.h
 * \brief Backward-compatible helper for parsing \c Q_ENUM cells out of a
 *        Blackchirp CSV.
 *
 * The header stands alone (no project includes) so the helper can be used
 * from both \c blackchirpcsv.h and \c headerstorage.h without creating a
 * circular include between the two.
 */

namespace BC::CSV {

/*!
 * \brief Decode a \c QVariant cell into a \c Q_ENUM-bearing enum, accepting
 *        both the enum name (current writer output) and the integer value
 *        (historical writer output) as input forms.
 *
 * \tparam E A \c Q_ENUM- or \c Q_ENUM_NS-registered enumeration type.
 * \param v Cell value as returned by \c BlackchirpCSV::readLine. Typically a
 *        \c QVariant wrapping a \c QString, or a metatype-tagged enum when
 *        the cell came from an in-memory pipeline rather than the disk.
 * \param defaultValue Value returned when neither form parses successfully.
 * \return The decoded enum value, or \a defaultValue when both forms fail.
 *
 * Resolution order:
 * -# Direct metatype hit — \c v already carries an \c E.
 * -# Numeric form — \c v.toInt() succeeds; the integer is reinterpreted as
 *    \c E without further validation against the meta-enum (matches the
 *    behavior of the historical \c static_cast call sites this helper
 *    replaces).
 * -# Name form — \c QMetaEnum::keyToValue accepts the cell text as an enum
 *    key.
 *
 * Use this helper at every read site that consumes a Blackchirp-written CSV
 * cell whose source was a \c Q_ENUM field. The dual-form invariant is
 * required because the on-disk representation of these cells changed from
 * numeric to name across versions, and Blackchirp must continue to read its
 * own historical output.
 */
template<typename E>
E enumFromVariant(const QVariant &v, E defaultValue)
{
    if(v.userType() == qMetaTypeId<E>())
        return v.value<E>();

    bool ok = false;
    int n = v.toInt(&ok);
    if(ok)
        return static_cast<E>(n);

    auto meta = QMetaEnum::fromType<E>();
    int idx = meta.keyToValue(v.toString().toUtf8().constData(),&ok);
    if(ok)
        return static_cast<E>(idx);

    return defaultValue;
}

}

#endif // ENUMCSVCONVERT_H
