import React from 'react';
import SearchArea from './SearchArea';
import SearchHeader from './SearchHeader';

export default function SearchPage(props) {
    return (
        <div>
            <SearchHeader />
            <SearchArea props={props}/>
        </div>
    )
}